## CODEX
- now: Gemma4 12B JANG4M media proof accounting was refreshed to consume current source proof instead of stale missing 20260607 artifact.
- proof: `tests/cross_matrix/run_full_release_objective_checklist.py` now points `GEMMA4_12B_JANG4M_MEDIA_SMOKE` at `build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json`; focused checklist tests passed `3/3`; regenerated `build/current-full-release-objective-checklist-after-gemma12-media-pointer-refresh-20260610.json` is still `status=open` but `failed_count` drops to `68` and `gemma4_12b_jang4m_media_artifact_exists/status_pass/media_rows_complete` are all green.
- boundary: this is proof accounting, not a new runtime load. Gemma4 12B JANG4M no-media exact code whitespace remains red, and the mixed-SWA native-cache row still records storage quantization disabled; Gemma E2B same-model tunnel parity also remains red. No release/PyPI/signing action.
- now: MiMo V2.5 JANGTQ_2 source video/audio E2E was rerun after the media-routing fix. Proof artifact: `build/current-mimo-v25-jangtq2-video-audio-source-proof-20260610.json`.
- proof: real source server loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` as MLLM, auto-enabled MiMo media, bound `visual=364`, `audio_encoder=75`, `speech_embeddings=20`, and assigned `459` media tensors. Video `/v1/chat/completions` with `video_url` reached mlx-vlm's numpy video reader and returned HTTP `200`; audio `input_audio` base64 decoded to wav and returned HTTP `200`.
- boundary: video transport/frame-through-vision is green, but semantic color answer is red: fixture first frame decodes as RGB `254,0,0`, yet model answered dominant color `black`; extracted red frame image and generated red/green/blue/white images also returned `Black.`. Icon text image still returns `VMLX`, so media path is active but visual semantics remain inconsistent. Audio returned `I hear a person saying "I'm fine."`, but transcript semantics are not independently verified.
- next other-agent action: rebuild/relaunch current Electron dev app from `b0d5bb5d` or newer and rerun MiMo JANGTQ_2 image/video/audio UI rows. Do not claim MiMo visual semantic quality green from solid-color fixtures; use source/dequant/reference embedding/logit comparison if visual quality must be fixed.
- now: MiMo V2.5 JANGTQ_2 source media-runtime routing was fixed and live-proven on the real local artifact. Proof artifact: `build/current-mimo-v25-jangtq2-media-runtime-source-proof-20260610.json`.
- proof: current source now auto-enables MiMo media only when the preserved bundle has local MiMo media runtime classes, media sidecars, token metadata, and indexed `visual.*` / `audio_encoder.*` / `speech_embeddings.*` weights. The real JANGTQ_2 skeleton constructs `visual` and `audio_encoder`, sets `_mimo_v2_bind_media_weights=True`, and binds `visual=364`, `audio_encoder=75`, `speech_embeddings=20`.
- live API: source server loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` as MLLM on port `8877`, assigned `459` media tensors, used native mixed full/SWA rotating cache, skipped generic TurboQuant KV by design, and recorded Metal baseline about `76.5 GB` active / `107.5 GB` max working set. `/v1/chat/completions` image data URL returned HTTP `200` with visible `The text "vMLX"`; the previous source image `400 unsupported media modality` row is cleared for source.
- cache/exactness boundary: repeated text chat hit paged cache with `cached_tokens=29`, `cache_detail=paged`, `ram_tokens_cached=56`; L2 disk was not enabled in this launch (`l2_tokens_on_disk=0`). Exactness remains red: text probe `Reply exactly: MIMO-OK` returned `MIMOOK`. Installed-app parity, video E2E, audio E2E, fresh-process L2 restore, and release clearance remain open.
- next other-agent action: rebuild/relaunch current Electron dev app from this source and rerun MiMo JANGTQ_2 image/video/audio UI rows; do not mask exactness with parser/string repair or call installed app green until rebuilt proof exists.
- now: MiMo V2.5 JANGTQ_2 A/B with `VMLINUX_DISABLE_MIMO_V2_SWITCHGLU_FAST_PATH=1` completed. Artifact `build/current-mimo-v25-jangtq2-exactness-variant-disable-vmlx-fastpath-20260610/result.json` has the same eight failed exactness rows as baseline; summary `build/current-mimo-v25-jangtq2-disable-vmlx-fastpath-boundary-20260610.json` records that vMLX's MiMo SwitchGLU fast path is not the primary cause.
- proof: disabled-fastpath outputs exactly match baseline mutations: `blue-cat -> blue` / `blue grass`, `B7-CAT-09 -> B7 CAT-09` / `B7CAT-09`, JSON/tool arguments mutate the same way. Tokenizer round-trip preserves `blue-cat`, `B7-CAT-09`, and `ACK-CB-742`, so tokenizer corruption is also excluded.
- still red: source-vs-quant remains unavailable. SSH to `erics-m5-max2.local` shows source checkpoint `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5` exists and is `294G`, but no source server is listening on `127.0.0.1:8126`. Remaining class is JANGTQ codebook/artifact/model-quality; next action is source/dequant first-divergence or rebuild/higher-fidelity artifact, not parser/cache/sampling masking.
- now: MiMo V2.5 JANGTQ_2 live no-source exactness variant probe completed on real local artifact. Artifact `build/current-mimo-v25-jangtq2-exactness-variant-live-20260610/result.json` is `status=open`; boundary summary `build/current-mimo-v25-jangtq2-exactness-variant-live-boundary-20260610.json` records the exact failures and runtime facts.
- proof: all eight exactness rows are red while protocol shape succeeds. Direct completions/chat/JSON/tool calls mutate hyphenated values: `blue-cat -> blue`, chat `blue-cat -> blue grass`, `B7-CAT-09 -> B7 CAT-09` or `B7CAT-09`, JSON `value` mutates to `blue`/`B7CAT-09` without hyphen, and required tool args mutate to `{"value":"blue"}` / `{"value":"B7CAT-09"}`. HTTP returned `200`, JSON parsed, and tool calls had `finish_reason=tool_calls`, so this is not a parser/JSON/tool-shape failure.
- runtime proof: real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded with `codec=turboquant_codebook`, `profile=JANGTQ_2`, `423` prestacked routed experts, trained `8/256` routed experts, native `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, paged cache and block L2 enabled, generic TurboQuant KV inactive by design, `807` RAM tokens cached, `807` L2 block tokens on disk, `10` block disk writes, active memory about `76812 MB`, peak about `78038 MB`.
- still red: source-vs-quant first-divergence remains unavailable because `http://erics-m5-max2.local:8126/health` timed out. Other-agent next action is bring up source/dequant reference and compare first divergence/logits/codebook/TurboQuantSwitchLinear selected-expert outputs for these exact literal probes. Do not mask with parser/JSON repair, string post-processing, sampling clamps, or cache changes.
- now: Qwen35 MXFP8-MTP same-model Responses raw SSE was freshly recaptured after the stricter parser/API/gateway contract. Artifact `build/current-responses-raw-sse-parity-qwen35-direct-gateway-source-vs-tunnel-after-strict-parser-contract-20260610.json` is `status=fail` only because the reused public tunnel SSE is stale/duplicate-index; current-source direct and current panel gateway are green under the stricter checks.
- proof: direct and gateway both preserve required tool args `{"value": "blue-cat"}`, keep reasoning enabled, complete the reasoning item lifecycle, emit final `response.output` order `[message, reasoning, function_call]`, keep final function arguments consistent with the stream, and use valid output indices `message=[0]`, `reasoning=[1]`, `function_call=[2]`. Gateway live capture also proves request kwargs were not dropped: `stream=true`, `max_output_tokens=512`, `temperature=0`, `top_p=1`, `top_k=0`, `enable_thinking=true`, `tool_choice=required`, `tool_count=1`, `first_tool_name=record_fact`.
- runtime proof: real `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP` loaded as `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`; native MTP active with tool requests capped to D1, hybrid 10 attention + 30 SSM cache active, live attention TurboQuant KV active, paged cache active, block L2 and SSM companion L2 enabled, first request wrote 4 blocks / 222 tokens, second gateway request hit paged cache with 222 tokens. Server RSS after health was about `35.77 GiB`, with about `79.18 GiB` system memory still available.
- still red: public tunnel parity is not cleared. The reused tunnel capture `build/responses-sse-captures-20260609/tunnel-qwen35-mxfp8-mtp-tool-recapture-max512-20260609.sse` still has duplicate output index `message=[0]`, `function_call=[0]`. Other-agent next action is rebuild/redeploy the public tunnel/backend from current source `841e5f40` or newer, then recapture the same model/request. Do not synthesize args, disable reasoning, drop kwargs, or call the stale tunnel green.
- now: MiMo V2.5 JANGTQ_2 exactness root-cause boundary artifact added at `build/current-mimo-v25-jangtq2-exactness-root-cause-boundary-20260610.json`. It consolidates current evidence that tokenizer roundtrip, chat template literal preservation, loader/runtime binding, parser/JSON repair, tool protocol shape, prefix/paged/L2/KV cache, continuous batching only, and hidden stochastic sampling are not the primary cause. Remaining class is artifact/logit/codebook/decode quality or replacement artifact contract.
- now: Cross-family parser/API/gateway streaming contract was tightened and source-fixed. Raw Responses SSE parity classification now tracks content deltas, reasoning item lifecycle, final `response.output` order/content/function arguments, and final-object consistency. The Qwen35 gateway live-capture harness now records request kwargs (`stream`, `max_output_tokens`, `temperature`, `top_p`, `top_k`, `enable_thinking`, `tool_choice`, tool count/name) so gateway proof can catch dropped kwargs, not just returned bytes.
- fix: `ensure_thinking_off_sentinel()` now preserves MiniMax tool-request open reasoning seed while still closing forced thinking-off prompts for LFM2 and Step3.7 tools. This fixes the parser seed mismatch found by the all-parser sweep: MiniMax tool prompts with `enable_thinking=False` must keep the planning rail open for interleaved reasoning/tool streaming instead of being mis-seeded as visible content.
- proof: focused parser/API sweep passed `385/385`: `tests/test_reasoning_tool_interaction.py tests/test_tool_parsers.py tests/test_reasoning_modes.py tests/test_streaming_reasoning.py tests/test_xml_function_tool_parser.py tests/test_gemma4_tool_parser.py tests/test_responses_raw_sse_parity_contract.py tests/test_qwen35_responses_raw_sse_capture.py`. `py_compile` passed for changed Python files. Panel gateway live-capture Vitest passed syntax/import in non-live mode with `1 skipped` as expected.
- now: Eric added a hard priority for auto tool usage, content deltas, reasoning deltas, interleaved reasoning/tool streaming, gateway/API parity, request kwargs, parser selection, and final-object consistency. This is now recorded in `.agents/CODEX_ACTIVE_DIRECTIVES_20260610.md` as a required cross-family parser/API/gateway contract. Do not synthesize tool args, disable reasoning, drop kwargs, or hide raw parser leaks after the fact. Test and fix all model reasoning/tool parser families.
- now: Active hard lane guard added at `.agents/CODEX_ACTIVE_DIRECTIVES_20260610.md`. Before any model load, proof run, release/package/PyPI/sign/notarize step, source edit, or commit, Codex must read that file and state the currently allowed lane in the status update. Every movement must be logged with request, action, command/artifact, proven/not-proven status, blockers, no-claims, and other-agent action.
- correction: Eric told this Codex instance not to work N2 JANG_1L ("im reamming nex n2 jang1l forget about it"). Codex then mistakenly launched an N2 JANG_1L proof. The runner and server were stopped, port `8876` was verified clear, and the unproven N2 JANG_1L source baseline patch was removed. Only `build/current-n2-jang1l-live-chat-cache-baseline-refresh-20260610.server.log` exists from that aborted run; there is no JSON proof artifact and it must not be used as evidence.
- hard boundary: N2 JANG_1L is Eric-owned until explicitly reassigned. Do not claim N2 JANG_1L fixed/proven/release-clear/cache-clear/tool-clear from partial or aborted work. Current allowed lanes for this Codex instance are MiMo V2.5 JANG/JANGTQ, Gemma JANG/MXFP/QAT, Qwen/Qwen3.6 Responses/tools/reasoning parity, N2 JANGTQ/non-JANG_1L only, and installed-app/release-surface proof only when explicitly appropriate.
- now: MiMo V2.5 JANGTQ_2 load-module contract was probed from current source after the dev-app exactness red row. The obvious runtime loader failure is excluded: top-level `config.quantization` is visible through MiMo `text_config`, q8 bookend tensors load as quantized modules, q4 attention affine tensors load as quantized modules, and routed experts load as 2-bit prestacked `TurboQuantSwitchLinear`.
- proof: `build/current-mimo-v25-jangtq2-load-module-contract-20260610.json` records `lm_head=QuantizedLinear(bits=8, group_size=64)`, `embed_tokens=QuantizedEmbedding(bits=8, group_size=64)`, layers 0/1/2/47 `self_attn.{qkv_proj,o_proj}=QuantizedLinear(bits=4, group_size=64)`, and layers 1/2/47 switch MLP projections as `jang_tools.turboquant.tq_kernel.TurboQuantSwitchLinear(bits=2)`. Load completed in about 7.0s using the vMLX MiMo runtime registration.
- classification: current evidence still points to MiMo JANGTQ_2 artifact/logit/decode quality, not parser/JSON repair, cache/L2, continuous batching, or missing sidecar binding. The current exactness red artifact remains `build/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-harness-assert-proof-20260610.json`: `ACK-CB-742 -> ACKCB-742`, `{"status":"ok","value":"blue-cat"} -> {"status":"ok","value":"blue"}`.
- next other-agent action: do not patch semantic values in parser/JSON repair or clamp sampling. Next useful proof is either a source-vs-quant first-divergence/logit probe or a rebuild/reupload of MiMo JANGTQ_2 with a literal-safe quantization contract; if runtime decode is suspected, compare TurboQuantSwitchLinear output/logits against dequant/reference for the same selected experts.
- boundary: no MiMo runtime patch was made because this probe did not find a source loader bug. MiMo JANGTQ_2 exactness remains red; MiMo media remains text-only/guarded; installed-app release clearance remains separate.
- now: Qwen35 same-model Responses raw SSE was recaptured after the source reasoning-output item fix. Current source direct and real panel gateway both preserve required tool args `{"value": "blue-cat"}` with reasoning enabled, no reasoning-disable workaround, and valid output indices `message=[0]`, `reasoning=[1]`, `function_call=[2]`.
- proof: `build/current-responses-raw-sse-parity-qwen35-direct-gateway-source-vs-tunnel-after-reasoning-item-index-20260610.json` is still `status=fail` only because the reused public tunnel capture remains stale with `message=[0]`, `function_call=[0]`. Direct SSE: `build/responses-sse-captures-20260610/direct-qwen35-mxfp8-mtp-tool-after-reasoning-item-index-20260610.sse`; gateway SSE: `build/responses-sse-captures-20260610/gateway-qwen35-mxfp8-mtp-tool-after-reasoning-item-index-20260610.sse`; server log: `build/responses-sse-captures-20260610/direct-qwen35-mxfp8-mtp-after-reasoning-item-index.server.log`.
- runtime proof: Qwen35 MXFP8-MTP loaded current source as `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`; native MTP was active, hybrid SSM cache was active, live attention TurboQuant KV was active, block disk L2 wrote 4 blocks/222 tokens, and the gateway request hit a paged+hybrid cache reuse path with 222 cached tokens. Memory after health was about 35.8 GiB RSS with 76.0 GiB still available.
- next other-agent action: rebuild/redeploy the public tunnel/backend from current source containing `841e5f40` or newer, then recapture Qwen35 tunnel raw SSE with the same request/model. Do not synthesize tool args, disable reasoning, add parser fallback injection, or treat the old tunnel SSE as current after the source/gateway recapture.
- boundary: this clears current-source direct+gateway reproduction of the #190/#192 Qwen tool-args/output-index failure, but deployed tunnel parity remains red until recaptured from a rebuilt tunnel. This does not touch Eric's N2 JANG_1L lane, MiMo exactness/media, Gemma audio semantics, installed-app proof, PyPI auth, or release signing.
- now: Gemma4 Unified JANG/MXFP audio false-advertisement is fixed in source. `_bundle_declares_native_audio()` now requires real `audio_tower.*` weights for `gemma4_unified` just like `gemma4`; `audio_config` plus `embed_audio.embedding_projection.weight` alone no longer advertises audio. The explicit experimental MXFP8 attention-FP16 override remains opt-in.
- proof: focused Gemma modality gate tests passed `7/7`; `py_compile` and `git diff --check` passed. Direct local path check now reports `['text', 'vision', 'video']` for `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`, `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-JANG_4M`, and `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4`, matching their projection-only audio tensors.
- boundary: this does not make Gemma audio work; it prevents fake audio routing and should turn those rows into honest unsupported-audio gates until a weight-backed audio tower bundle is present and live-proven.
- now: Responses streaming reasoning/tool output-index bookkeeping has a source fix. `stream_responses_api()` now emits a real `reasoning` output item lifecycle for `response.reasoning_summary_text.*` streams and reserves index `1` for it, so subsequent `function_call` items move to index `2` instead of colliding with message/reasoning index `0/1`. No tool arguments are synthesized or repaired by this change.
- proof: focused source checks passed: `tests/test_server.py -k streaming_responses_tool_call...` selected `5/5`; `tests/test_responses_raw_sse_parity_contract.py tests/test_hybrid_batching.py -k raw_sse_parity...` selected `19/19`; `py_compile vmlx_engine/server.py` and `git diff --check` passed. Direct in-process SSE probe showed event indices `message=0`, `reasoning=1`, `function_call=2`; final `response.output` order `[message, reasoning, function_call]`; function arguments done at index `2`.
- boundary: source/in-process proof only. Still need same-model direct/gateway/tunnel raw SSE recapture from the live routes before closing #190/#192 deployed parity. This does not touch N2 JANG_1L, MiMo artifact exactness, Gemma audio semantics, installed-app proof, or PyPI auth.
- now: Fresh public checkpoint release `v1.5.57` is signed, notarized, uploaded, and live on GitHub/updater/site. This supersedes the stale `v1.5.56` public release surface while preserving the broader release-manifest truth that the full model matrix is not production-green.
- source: committed `52742f40 Release vMLX 1.5.57 checkpoint`; pushed to `origin/main` and `origin/codex/pr-intake-manifest`; tag `v1.5.57` pushed.
- GitHub releases: `jjang-ai/vmlx` and `jjang-ai/mlxstudio` both have fresh `v1.5.57` releases with all four assets. Published times: `vmlx` `2026-06-10T09:27:37Z`; `mlxstudio` `2026-06-10T09:29:43Z`.
- notarization/signing: Sequoia DMG notarization id `78aeae6b-14c1-4c14-8da5-d428908b2079`; Tahoe DMG notarization id `08e020a1-0b93-43c3-b3e9-9fd6bd83e723`. `panel/scripts/verify-release-dmgs.sh` passed for both: valid DMG checksum, valid Developer ID signature, stapled ticket, stapler validation, and Gatekeeper `source=Notarized Developer ID`.
- release hashes: Sequoia DMG `68bfea742acf991887d7cf6858a1300fd20c26b3cd49f0e6601d88506f379525`; Tahoe DMG `bafe3e4329b341f993ee0c25e34c0bef78131e90066a8b72f3a5ead8b6160ef7`; Sequoia blockmap `dedbbce28ee509210aa067345f5b3641ab9dbd7aa714247d9e595154e3ce89eb`; Tahoe blockmap `533d5eccc4d829db018b514672a916232e46f2ddd0028030aaf5368081c8a019`.
- updater/site: `jjang-ai/mlxstudio@f1d7a5c` updates `latest.json` to `1.5.57`; live `https://mlx.studio/update/latest.json` returns `1.5.57` with matching hashes; live download page displays `1.5.57` and matching Sequoia/Tahoe hashes; Cloudflare purge for `mlx.studio` succeeded.
- release-surface proof: `build/current-release-surface-contract-after-1.5.57-release-20260610.json` is green for source version consistency, local updater validity, public GitHub release/assets/tag, raw updater, site updater/cache headers, and staged-source checks. It is red only on `public_pypi_has_release_files=false`.
- PyPI status: local `/tmp/vmlx-dist-1.5.57` wheel/sdist built and `twine check` passed. GitHub PyPI workflow failed twice because trusted publishing has no matching PyPI publisher for `repo:jjang-ai/vmlx:environment:pypi` (main and tag claims both invalid). Direct `twine upload` using existing `~/.pypirc` also failed `403 Forbidden`. PyPI still reports `vmlx==1.5.56`; no partial `1.5.57` landed. Need rotated/authorized PyPI token in GitHub secret `PYPI_API_TOKEN` or PyPI trusted publisher configured for the shown claims before PyPI can be completed.
- checkpoint boundary: this is a signed/notarized checkpoint release for users, not a claim that open N2/MiMo/Gemma media/Responses/full installed-app rows are all green. Eric is handling N2 JANG_1L separately; do not duplicate that lane unless asked.
- now: N2 JANGTQ2 real dev-app Responses/tool/cache proof is now consumed by the objective digest/checklist/release-manifest pointer chain. Release remains red, but the checkpoint-candidate N2 JANGTQ2 proof is no longer only sitting in handoff docs.
- proof: `build/current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json`, `build/current-full-release-objective-checklist-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json`, and `build/current-release-regression-manifest-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json`.
- proven consumed row: `jangtq2_real_ui_prevresp_proof.status=pass`, real Electron dev app, `/v1/responses`, built-in tool loop, exact probe files for `REAL_UI_LIVE_TOOL_ONE/TWO`, native `hybrid_ssm_v1`, live attention TurboQuant KV, `cache_hit_tokens=17083`, `l2_tokens_on_disk=20662`, block L2 hits/writes, and SSM disk hits/stores.
- still red: checklist remains `status=open`, `failed_count=73`; manifest remains `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`. No release action in this proof-consumption commit.
- now: MiMo V2.5 JANG_2L release-manifest root-cause accounting now consumes current 20260610/20260609 proof artifacts instead of stale missing 20260606 pointers. Release remains red.
- proof: regenerated `build/current-release-regression-manifest-after-mimo-jang2l-responses-l2-accounting-20260610.json`; command exited `1` because `current_proof_sweep=fail`, `prepackage_ready=false`, and `release_ready=false`, but JSON validation passed and `current_proof_sweep.mimo_v2_jang2l_root_cause.missing=[]`.
- proven by the manifest row: MiMo JANG_2L metadata honesty is green, narrow text/cache is green, SwitchGLU selected-expert parity is green, direct Chat tool protocol is not the current blocker, fresh-process block-disk L2 restore is green, and Responses transport/deltas/`previous_response_id`/cache/L2 are green.
- still red in the same row: long-prompt first-request Metal OOM, JANGTQ_2 artifact exactness, decode speed, media wiring, JANG_2L live media/L2, JANG_2L Responses/tool semantic drift, and source-vs-quant/no-source classification boundary. No package/sign/notarize/tag/upload/release action.
- now: MiMo V2.5 JANG_2L current Electron dev-app Responses/tools rerun completed and is still release-red, but the blocker is now classified as tool-loop semantics instead of CDP timeout.
- proof: `build/current-real-ui-live-model-mimo-v25-jang2l-responses-tools-rerun-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-live-model-mimo-v25-jang2l-responses-tools-rerun-20260610-proof.json`; screenshot is `docs/internal/agent-notes/current-real-ui-live-model-mimo-v25-jang2l-responses-tools-rerun-20260610-chat.png`.
- proven: real Electron dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` load, `/v1/responses`, Responses delta streaming, `previous_response_id` tool-result follow-up, two completed visible assistant turns, generation defaults `temperature=0`, `top_p=1`, `max_tokens=384`, native `mixed_swa_kv_v1`, paged cache, and block L2.
- runtime/cache: active memory `105485.6 MB`, peak `110316.4 MB`, `cache_hit_tokens=4552`, last cache `paged` / `3481` tokens / `reconstruction_ok=true`, `l2_block_tokens_on_disk=4960`, block disk `disk_hits=36`, `disk_writes=80`.
- red: release assertion failed because no `long_tool_loop` surface was recorded; visible content drifted `REAL_UI_LIVE_TOOL_ONE` to `REAL_UI_LAND_TOOL_ONE`, and tool/file semantics did not satisfy the proof contract. This does not clear MiMo JANG_2L Responses/tools or release support. No release action.
- now: MiMo V2.5 JANG_2L fresh-process block-disk L2 restore is source-green.
- proof: `build/current-mimo-v25-jang2l-restart-l2-restore-20260610-rerun/summary.json`, `status=pass`; per-model result is `build/current-mimo-v25-jang2l-restart-l2-restore-20260610-rerun/MiMo-V2.5-JANG_2L/result.json`.
- proven: two fresh `vmlx_engine.cli serve` processes loaded real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` on the 128GB host with paged cache and `--enable-block-disk-cache`; first process wrote one block / `48` tokens to block L2; second process opened the existing store, returned HTTP `200`, and restored `48` cached tokens with `cache_detail=paged+disk`, block disk `disk_hits=1`, scheduler cache `disk_hits=1`, and `reconstruction_ok=true`.
- runtime/cache: JANG v2 loaded by mmap in about `14s`, wired limit `115 GB`, active Metal baseline about `102.5 GB`, native cache `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, components `full_attention_kv`, `sliding_window_kv`, and `rotating_window_metadata`; generic TurboQuant KV stayed inactive by design for MiMo.
- boundary: this clears MiMo JANG_2L source fresh-process L2 restore only. The same artifact still marks visible exact `ACK` as review, and MiMo JANG_2L tool-loop/Responses/media/installed-app/public tunnel/release rows remain open. No package, signing, notarization, tag, upload, or release action.
- now: Release gate/checklist proof accounting is refreshed after the MiMo/N2 dev-app proofs; it still blocks release.
- proof: canonical objective digest is now `build/current-objective-proof-after-mimo-n2-dev-app-proof-refresh-20260610.json`; full checklist is `build/current-full-release-objective-checklist-after-mimo-n2-dev-app-proof-refresh-20260610.json`; release manifest is `build/current-release-regression-manifest-after-mimo-n2-dev-app-proof-refresh-20260610.json`.
- evidence: N2 Pro 397B row now includes current-source forced JANG_1L proof plus real Electron dev-app bounded/one-turn proofs instead of relying on the stale memory-only proof path.
- result: objective digest remains `open`; checklist remains `status=open`, `failed_count=73`; release manifest remains `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- boundary: proof-pointer/gate refresh only. No package, sign, notarize, tag, appcast, upload, PyPI, or public release action.
- now: MiMo JANGTQ_2 current Electron dev-app exactness is red with raw harness assertions, not just compact post-classification.
- proof: `build/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-harness-assert-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-harness-assert-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-harness-assert-20260610-chat.png`.
- harness: `panel/scripts/live-real-ui-model-proof.mjs` now supports `VMLINUX_REAL_UI_EXPECT_ASSISTANT_1` and `VMLINUX_REAL_UI_EXPECT_ASSISTANT_2`, and fails raw real-UI proofs on visible assistant exact mismatch.
- evidence: real dev app, real JANGTQ_2 load, two completed UI turns, expected `ACK-CB-742` became `ACKCB-742`, expected `{"status":"ok","value":"blue-cat"}` became `{"status":"ok","value":"blue"}`.
- runtime/cache: `codec=turboquant_codebook`, `profile=JANGTQ_2`, prestacked routed experts `423`, native `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, generic TurboQuant KV inactive, paged cache hit, `cache_hit_tokens=40`, `l2_block_tokens_on_disk=114`, `l2_tokens_on_disk=114`, block-disk writes `3`.
- boundary: this strengthens the artifact/logit/codebook/decode exactness blocker; do not mask with parser/JSON repair, sampling clamps, or cache changes. No release action.
- now: N2 JANG_1L current Electron dev-app one-turn visible-output probe is red, but it separates first-turn whitespace output from the prior second-turn Metal guard.
- proof: `build/current-real-ui-dev-app-n2-jang1l-one-turn-visible-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-n2-jang1l-one-turn-visible-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-n2-jang1l-one-turn-visible-20260610-chat.png`.
- harness: `panel/scripts/live-real-ui-model-proof.mjs` now supports `VMLINUX_REAL_UI_SECOND_TURN=0`; artifacts record `secondTurnEnabled=false`. This is for bounded one-turn diagnosis only and does not claim multi-turn/cache reuse.
- evidence: real dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANG_1L` load, one Chat request completed with no renderer send error or Metal 503, 16 stream events and 1 complete event, but visible assistant content persisted empty; stream trace was whitespace only.
- runtime/cache: qwen3_5_moe/JANG_1L, qwen tool parser, qwen3 reasoning parser, native `hybrid_ssm_v1`, live attention TurboQuant KV, SSM companion state, async rederive, paged cache, block L2, SSM companion L2, `ram_tokens_cached=19`, `l2_block_tokens_on_disk=19`, `l2_ssm_tokens_on_disk=19`, `l2_tokens_on_disk=38`, active memory `112550.2 MB`, peak `112807.4 MB`.
- boundary: N2 JANG_1L remains not release-clear: visible output, second-turn/cache reuse, tools, Responses, L2 restart, media, installed-app parity, package/sign/notarize all remain open. No release action.
- now: MiMo V2.5 JANG_2L current Electron dev-app audio is classified red by an honest text-only runtime guard.
- proof: `build/current-real-ui-dev-app-mimo-v25-jang2l-audio-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jang2l-audio-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jang2l-audio-20260610-chat.png`.
- evidence: real dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` load, two visible text turns before audio, server `MEDIA_DIAG` saw `input_audio`, then `/v1/chat/completions` returned HTTP `400`: `unsupported media modality audio because the loaded runtime is text-only. Supported modalities: text.`
- runtime/cache: active memory `105016.1 MB`, peak `106151.0 MB`, `codec=affine_quantized_matmul`, `profile=JANG_2L_322_D3E16`, Metal NA eligible, native `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, generic TurboQuant KV correctly inactive, `ram_tokens_cached=110`, `l2_block_tokens_on_disk=110`, `l2_tokens_on_disk=110`, block-disk writes `3`.
- boundary: this does not clear MiMo JANG_2L audio/media support, installed-app parity, exactness/tool-loop issues, package/sign/notarize, or release support. No release action.
- now: MiMo V2.5 JANG_2L current Electron dev-app video is classified red by an honest text-only runtime guard.
- proof: `build/current-real-ui-dev-app-mimo-v25-jang2l-video-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jang2l-video-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jang2l-video-20260610-chat.png`.
- evidence: real dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` load, two visible text turns before video, server `MEDIA_DIAG` saw `video_url`, then `/v1/chat/completions` returned HTTP `400`: `unsupported media modality video because the loaded runtime is text-only. Supported modalities: text.`
- runtime/cache: active memory `105016.1 MB`, peak `106152.4 MB`, `codec=affine_quantized_matmul`, `profile=JANG_2L_322_D3E16`, Metal NA eligible, native `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, generic TurboQuant KV correctly inactive, `ram_tokens_cached=110`, `l2_block_tokens_on_disk=110`, `l2_tokens_on_disk=110`, block-disk writes `3`.
- boundary: this does not clear MiMo JANG_2L video/media support, installed-app parity, exactness/tool-loop issues, package/sign/notarize, or release support. No release action.
- now: MiMo V2.5 JANGTQ_2 current Electron dev-app audio is classified red by an honest text-only runtime guard.
- proof: `build/current-real-ui-dev-app-mimo-v25-jangtq2-audio-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-audio-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-audio-20260610-chat.png`.
- evidence: real dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` load, two visible text turns before audio, server `MEDIA_DIAG` saw `input_audio`, then `/v1/chat/completions` returned HTTP `400`: `unsupported media modality audio because the loaded runtime is text-only. Supported modalities: text.`
- runtime/cache: active memory `76491.8 MB`, peak `77127.3 MB`, `codec=turboquant_codebook`, `profile=JANGTQ_2`, prestacked routed experts `423`, native `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, generic TurboQuant KV correctly inactive, `ram_tokens_cached=132`, `l2_block_tokens_on_disk=132`, `l2_tokens_on_disk=132`, block-disk writes `3`.
- boundary: this does not clear MiMo JANGTQ_2 audio/media support, installed-app parity, exactness, package/sign/notarize, or release support. No release action.
- now: MiMo V2.5 JANGTQ_2 current Electron dev-app video is classified red by an honest text-only runtime guard.
- proof: `build/current-real-ui-dev-app-mimo-v25-jangtq2-video-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-video-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-video-20260610-chat.png`.
- evidence: real dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` load, two visible text turns before video, server `MEDIA_DIAG` saw `video_url`, then `/v1/chat/completions` returned HTTP `400`: `unsupported media modality video because the loaded runtime is text-only. Supported modalities: text.`
- runtime/cache: active memory `76491.8 MB`, peak `77127.3 MB`, `codec=turboquant_codebook`, `profile=JANGTQ_2`, prestacked routed experts `423`, native `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, generic TurboQuant KV correctly inactive, `ram_tokens_cached=132`, `l2_block_tokens_on_disk=132`, `l2_tokens_on_disk=132`, block-disk writes `3`.
- boundary: this does not clear MiMo JANGTQ_2 video/media support, installed-app parity, exactness, package/sign/notarize, or release support. No release action.
- now: N2 JANG_1L current Electron dev-app bounded Chat was attempted. It is partial/red: real load and one HTTP `200` Chat request are proven; visible quality and second-turn/cache reuse are red.
- proof: `build/current-real-ui-dev-app-n2-jang1l-bounded-chat-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-n2-jang1l-bounded-chat-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-n2-jang1l-bounded-chat-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANG_1L` load, `/health`, first Chat Completions HTTP `200`, qwen3_5_moe/JANG_1L, qwen tool parser, qwen3 reasoning parser, native `hybrid_ssm_v1`, live attention TurboQuant KV, SSM companion state, async rederive, paged cache, block L2, and SSM companion L2.
- runtime/cache: active memory `112548.8 MB`, peak `112803.4 MB`, `actual_bits=2.13`, `profile=JANG_1L`, `prestacked_switch=540`, 15 attention TQ-KV layers, 45 SSM companion layers, `ram_tokens_cached=18`, `l2_block_tokens_on_disk=18`, `l2_ssm_tokens_on_disk=18`, `l2_tokens_on_disk=36`, block-disk writes `1`, SSM companion disk store `1`.
- red: first assistant visible content was empty/whitespace; second UI turn returned HTTP `503` from Metal working-set guard at `102%` of the `107.5GB` cap. This does not clear N2 JANG_1L release support. No release action.
- now: Gemma 4 31B QAT JANG4M current Electron dev-app audio is classified red by an honest unsupported-modality guard.
- proof: `build/current-real-ui-dev-app-gemma4-31b-jang4m-audio-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-audio-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-audio-20260610-chat.png`.
- evidence: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` load, two visible text turns before audio, app forced multimodal for one audio file, server `MEDIA_DIAG` saw `input_audio`, then `/v1/chat/completions` returned HTTP `400`: `unsupported media modality audio. Supported modalities: text, vision, video.`
- runtime/cache: active memory `25324.6 MB`, peak `25728.6 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, Metal NA active, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=62`, `l2_tokens_on_disk=62`, block-disk writes `2`.
- boundary: this does not invalidate Gemma 31B text/tools/image/video/cache rows, but audio support remains red and must not be claimed. No release action.
- now: Gemma 4 26B A4B QAT JANG4M current Electron dev-app audio is classified red by an honest unsupported-modality guard.
- proof: `build/current-real-ui-dev-app-gemma4-26b-jang4m-audio-proof-20260610.json`, `status=fail`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-audio-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-audio-20260610-chat.png`.
- evidence: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` load, two visible text turns before audio, app forced multimodal for one audio file, server `MEDIA_DIAG` saw `input_audio`, then `/v1/chat/completions` returned HTTP `400`: `unsupported media modality audio. Supported modalities: text, vision, video.`
- runtime/cache: active memory `17648.7 MB`, peak `17843.1 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, Metal NA active, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=64`, `l2_tokens_on_disk=64`, block-disk writes `2`.
- boundary: this does not invalidate Gemma 26B text/tools/image/video/cache rows, but audio support remains red and must not be claimed. No release action.
- now: N2 JANG_1L was force-loaded after the Gemma video proofs instead of stopping at preflight. It is partially green for load + one bounded Chat request, still red for cache reuse/tool/Responses/L2 release clearance.
- proof: `build/current-n2-jang1l-live-chat-cache-forced-after-gemma-video-20260610.json` is `status=fail`; server log is `build/current-n2-jang1l-live-chat-cache-forced-after-gemma-video-20260610.server.log`.
- proven: real `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANG_1L` load, `/health` reached, `qwen3_5_moe`, `profile=JANG_1L`, `actual_bits=2.13`, `prestacked_switch=540`, `trained_active_experts=10`, qwen tool parser, qwen3 reasoning parser, native `hybrid_ssm_v1`, live attention TurboQuant KV for 15 attention layers, SSM companion state for 45 layers, async rederive, paged cache, block L2, SSM companion L2, and one bounded Chat Completions request returned HTTP `200`.
- red: cache warm and cache hit requests returned HTTP `503` because the Metal working-set guard rejected at `102%` of the `107.5GB` cap after the first request. The first response body had empty visible text for the 8-token bounded probe, so this is not quality clearance.
- memory: before launch `114.03 GiB` available, after health `112.92 GiB`, after first request `6.41 GiB`; final health active Metal `112357.6 MB`, peak `112563.1 MB`.
- boundary: this proves the model can load and serve one bounded Chat request on the 128GB Mac. It does not clear N2 JANG_1L cache reuse, tools, Responses, stream, L2 restart, media, UI installed-app, or release support. No release action.
- now: Gemma 4 31B QAT JANG4M current Electron dev-app video/VL row is green with explicit `max_prompt_tokens=12000`.
- proof: `build/current-real-ui-dev-app-gemma4-31b-jang4m-video-proof-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-video-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-video-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` load, app-persisted `video_url`, server `MEDIA_DIAG` with `video_url`, base64 MP4 decode, `4` extracted frames, Gemma media fallback through vision frames, visible semantic answer `The provided image is a solid red square.`, no raw parser/reasoning leak, no persisted tools/reasoning, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `25842.8 MB`, peak `26233.1 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, Metal NA active, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=62`, `l2_tokens_on_disk=62`, block-disk writes `2`, video-turn media-prefix cache stored `355` prompt tokens.
- boundary: this does not clear default-4k video behavior, 31B audio, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 26B A4B QAT JANG4M current Electron dev-app video/VL row is green with explicit `max_prompt_tokens=12000`.
- proof: `build/current-real-ui-dev-app-gemma4-26b-jang4m-video-proof-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-video-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-video-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` load, app-persisted `video_url`, server `MEDIA_DIAG` with `video_url`, base64 MP4 decode, `4` extracted frames, Gemma media fallback through vision frames, visible semantic answer `The video is a solid, static red square. REAL_UI_LIVE.`, no raw parser/reasoning leak, no persisted tools/reasoning, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `17779.1 MB`, peak `18557.2 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=64`, `l2_tokens_on_disk=64`, block-disk writes `2`, video-turn media-prefix cache stored `357` prompt tokens.
- boundary: this does not clear default-4k video behavior, 26B audio, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 31B QAT JANG4M current Electron dev-app image/VL row is green.
- proof: `build/current-real-ui-dev-app-gemma4-31b-jang4m-image-proof-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-image-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-image-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` load, app-persisted image attachment, server `MEDIA_DIAG` with `image_url`, Gemma media fallback with `1 image(s)`, visible semantic answer `Red`, no raw parser/reasoning leak, no persisted tools/reasoning, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `25850.6 MB`, peak `26233.1 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=62`, `l2_tokens_on_disk=62`, block-disk writes `2`, image-turn media-prefix cache stored `365` prompt tokens.
- boundary: this does not clear 31B video/audio, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 26B A4B QAT JANG4M current Electron dev-app image/VL row is green.
- proof: `build/current-real-ui-dev-app-gemma4-26b-jang4m-image-proof-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-image-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-image-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` load, app-persisted image attachment, server `MEDIA_DIAG` with `image_url`, Gemma media fallback with `1 image(s)`, visible semantic answer `Red`, no raw parser/reasoning leak, no persisted tools/reasoning, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `17780.6 MB`, peak `18557.2 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=64`, `l2_tokens_on_disk=64`, block-disk writes `2`, image-turn media-prefix cache stored `367` prompt tokens.
- boundary: this does not clear 26B video/audio, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 26B A4B QAT JANG4M current Electron dev-app Responses built-in tool/cache row is green.
- proof: `build/current-real-ui-dev-app-gemma4-26b-jang4m-responses-tools-cache-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-responses-tools-cache-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-responses-tools-cache-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` load, `/v1/responses`, built-in `run_command`, `previous_response_id` plus `function_call_output` follow-ups, two completed visible assistant turns, exact probe files `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`, Responses delta/cache-detail surfaces, no raw parser/reasoning leak, Gemma4 parser family, mixed-SWA paged cache, and block L2 writes/hits.
- runtime/cache: active memory `17782 MB`, peak `19943.9 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=3538`, `l2_block_tokens_on_disk=3559`, `l2_tokens_on_disk=3559`, block-disk hits `30`, block-disk writes `58`.
- boundary: this does not clear 26B media, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 31B QAT JANG4M current Electron dev-app Responses built-in tool/cache row is green.
- proof: `build/current-real-ui-dev-app-gemma4-31b-jang4m-responses-tools-cache-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-responses-tools-cache-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-responses-tools-cache-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` load, `/v1/responses`, built-in `run_command`, `previous_response_id` plus `function_call_output` follow-ups, two completed visible assistant turns, exact probe files `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`, Responses delta/cache-detail surfaces, no raw parser/reasoning leak, Gemma4 parser family, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `28090.3 MB`, peak `34587.3 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=320`, `l2_block_tokens_on_disk=1960`, `l2_tokens_on_disk=1960`, block-disk writes `58`, block-disk evictions `26`.
- boundary: this does not clear 31B media, installed-app parity, public tunnel SSE, 26B tools/media, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 31B QAT JANG4M current Electron dev-app exact-output row is green for no-media Chat Completions.
- proof: `build/current-real-ui-dev-app-gemma4-31b-jang4m-exact-output-proof-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-exact-output-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-31b-jang4m-exact-output-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` load, exact `GEMMA31-JANG4M-ACK-742`, exact `{"status":"ok","value":"gemma31-jang4m-blue"}`, no raw parser/reasoning leak, no persisted tools/reasoning, Gemma4 parser family, JANG affine Metal NA eligibility, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `25333.1 MB`, peak `25778.7 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=29`, `l2_block_tokens_on_disk=81`, `l2_tokens_on_disk=81`, block-disk writes `2`.
- boundary: this does not clear 31B tools/Responses/media, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: Gemma 4 26B A4B QAT JANG4M current Electron dev-app exact-output row is green for no-media Chat Completions.
- proof: `build/current-real-ui-dev-app-gemma4-26b-jang4m-exact-output-proof-20260610.json`, `status=pass`; raw proof is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-exact-output-20260610-proof.json`, screenshot is `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-26b-jang4m-exact-output-20260610-chat.png`.
- proven: real dev app, real `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` load, exact `GEMMA26-JANG4M-ACK-742`, exact `{"status":"ok","value":"gemma26-jang4m-blue"}`, no raw parser/reasoning leak, no persisted tools/reasoning, Gemma4 parser family, JANG affine Metal NA eligibility, mixed-SWA paged cache, and block L2 writes.
- runtime/cache: active memory `17650.5 MB`, peak `17842.9 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=29`, `l2_block_tokens_on_disk=81`, `l2_tokens_on_disk=81`, block-disk writes `2`.
- boundary: this does not clear 26B tools/Responses/media, 31B, installed-app parity, public tunnel SSE, package/sign/notarize/tag/upload/download, or full release readiness. No release action.
- now: N2 JANGTQ2 default real dev-app Responses tool/cache/delta path is green after a scoped panel Responses follow-up fix. The panel now sends in-turn tool-result follow-ups as `function_call_output` input with `previous_response_id` and does not re-apply the original explicit tool choice on that follow-up.
- proof: `build/current-real-ui-live-model-n2-jangtq2-dev-app-prevresp-proof-20260610.json`, `status=pass`; raw ignored proof is `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-prevresp-default-20260610-proof.json`. The app log records two `Responses tool follow-up using previous_response_id=... with 1 function_call_output item(s)` lines.
- proven: real Electron dev app, real 101 GiB N2 JANGTQ2 load, `/v1/responses`, built-in `run_command` loop, tool-result continuation, two visible assistant turns (`REAL_UI_LIVE_TOOL_ONE`, `REAL_UI_LIVE_TOOL_TWO`), renderer content deltas (`count=8`, `count=15`), hybrid SSM cache, attention-only TurboQuant KV, server cache controls, block L2 and SSM disk (`l2_tokens_on_disk=20662`, `block_disk_hits=110`, `ssm_disk_hits=1`).
- boundary: a stricter custom prompt is still red in `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-prevresp-longdelta-20260610-proof.json` with repeated `!` output and missing second tool file after a tool-choice-required error. Installed-app parity, N2 media, public tunnel parity, N2 JANG_1L, and release gates remain open. No package/sign/notarize/tag/upload/download action.
- now: Gemma 4 12B JANG4M real dev-app audio is classified red with a concrete boundary. The harness now supports `VMLINUX_REAL_UI_CHECK_AUDIO`; the app persisted `input_audio`, the server decoded the generated WAV, and the model streamed visible output, but it did not transcribe `audio present`.
- proof: `build/current-real-ui-live-model-gemma4-12b-jang4m-audio-proof-20260610.json`, `status=fail`; raw ignored proof is `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-audio-cache-20260610-proof.json`. Positive surfaces include current Electron dev app, real loaded Gemma JANG4M model, Chat Completions, `persistedAudioAttachment=true`, server WAV decode, server cache controls, `cache_detail=paged+mixed_swa`, `cacheHitTokens=67`, `l2_tokens_on_disk=67`, and `disk_writes=2`.
- boundary: this is not an app attachment-loss or cache/L2 failure. Gemma JANG4M image and video have separate green dev-app rows; audio semantic E2E remains red and must not be claimed in a checkpoint release. No package/sign/notarize/tag/upload/download action.
- now: Gemma 4 12B JANG4M dev-app video is live-proven with an explicit context boundary. Default 4k prompt cap failed with `HTTP 413 prompt_too_long` at about `8315` prompt tokens; rerun with `max_prompt_tokens=12000` passed.
- proof: `build/current-real-ui-live-model-gemma4-12b-jang4m-video-proof-20260610.json`, `status=pass`; raw ignored proof is `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-video-cache-max12k-20260610-proof.json`. A 1-second 64x64 solid-red MP4 was attached in the real Electron dev app as `video_url`, decoded by the server, frame-extracted, and answered as a solid/dark red screen.
- cache/runtime evidence: Gemma native mixed-SWA cache stayed active with prefix+paged+block-L2; `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=84`, `disk_writes=2`. Generic TurboQuant KV remains correctly inactive for Gemma mixed-SWA.
- boundary: Gemma 12B JANG4M dev-app image and video are now proven; Gemma audio semantic E2E, installed-app parity, public tunnel SSE, and release package/sign/notarize/tag/download remain open. Do not claim video works at the 4k context cap.
- now: N2 JANGTQ2 dev-app Responses content-delta surface is green in a separate real Electron app proof. This complements the earlier N2 built-in-tool/cache app proof; it does not replace it.
- proof: `build/current-real-ui-live-model-n2-jangtq2-dev-app-delta-proof-20260610.json`, `status=pass`; raw ignored proof is `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-delta-only-20260610-proof.json`. The run loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`, used `/v1/responses`, and recorded `responses_delta_streaming`, `responses_cache_detail_usage`, `cache_hit_telemetry`, `native_cache_status`, `l2_disk_storage`, and server cache controls.
- streaming/cache evidence: two assistant stream traces had `count=21` and `count=24`; visible content carried `N2_APP_DELTA_ONE` and `N2_APP_DELTA_TWO`; second trace had `cached_tokens=45`, `cache_detail=paged+ssm`; final cache totals included `l2_block_tokens_on_disk=120`, `l2_ssm_tokens_on_disk=274`, `l2_tokens_on_disk=394`.
- boundary: N2 app tool loop/cache and N2 app content-delta are now proven in separate app runs. The first post-tool visible answer quality in the tool-loop run remains a known red surface; installed-app parity, media, public tunnel SSE parity, package/sign/notarize/tag/download remain open. No release action.
- now: MiMo JANG_2L real-app tool boundary is reclassified after a scoped panel request fix. Source change pins `tool_choice` only when the latest user message explicitly names exactly one available built-in tool; Chat and Responses request-builder coverage proves the shape, while no-name and multi-name prompts remain unpinned.
- proof: direct MiMo JANG_2L server probe with only `run_command` is green (`build/current-mimo-v25-jang2l-chat-tool-boundary-20260610.json`); full panel tool surface reproduces the failure (`HTTP 413` / `prompt_too_long` at `4754` tokens with max `4096`) but specific/required `run_command` still works (`build/current-mimo-v25-jang2l-chat-tool-boundary-fulltools-20260610.json`).
- app proof: post-fix real Electron dev-app runs loaded the 105 GiB MiMo JANG_2L row and executed `run_command` calls with paged cache/L2 positive, but remain `status=fail` in `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-after-toolchoice-proof-20260610.json`. The model mutated requested tool semantics (`REAL_UI_LIVE_TOOL_ONE` -> `REAL_UI_LAND_TOOL_ONE`, expected working-directory probe files -> `/tmp/real_ui_land_tool_*.txt`), and a stricter exact-command prompt produced empty/partial visible output.
- validation: `panel/tests/request-builder.test.ts` passed `68/68`; focused request/model registry tests passed `8/8`; `panel` typecheck passed; MiMo boundary probe py_compile passed. Boundary: MiMo app `long_tool_loop`, Responses tool path, fresh-process L2 restore, media, installed-app parity, and release readiness remain red/open. No package/sign/notarize/tag/download action.
- now: reduced `api/ui` launch-parity blocker for the Gemma/MiMo/N2 checkpoint lane. Fixed panel autodetect drift and regenerated proof artifacts.
- fix: `panel/src/main/model-config-registry.ts` now maps `gemma4_unified` / `gemma4_unified_text` to Gemma4 so exact local Gemma 12B MXFP4/JANG4M bundles no longer fall to `unknown`; panel now exposes Gemma4 parsers and rotating/paged mixed-SWA cache for those bundles.
- fix: panel MiMo detection now matches Python runtime policy: `cacheSubtype=mimo_v2_asymmetric_swa`, `usePagedCache=true`, `toolParser=xml_function`, no automatic reasoning claim until visible-final thinking proof exists. Runtime and UI cache-subtype paged guards now include `mimo_v2_asymmetric_swa`.
- proof: `build/current-panel-settings-contract-proof-20260610-mimo-n2-gemma-launch-parity.json`, `status=pass`, `missing_source_markers=[]`; panel settings contract passed 315 tests, panel model registry passed 66, engine model registry passed 140, CLI flag contract passed 9, and panel typecheck passed.
- proof: `build/current-panel-exact-local-model-detect-mimo-n2-gemma-20260610.json`, `status=pass`; exact local Gemma, MiMo, and N2 checkpoint directories detect as expected. N2 JANG_1L is explicitly `forceTextOnly=true`, not claimed VL-ready.
- validation: focused tests passed `panel/tests/model-config-registry.test.ts` 66/66, `panel/tests/settings-flow.test.ts` 254/254, and `tests/test_panel_cli_flag_contract.py` 9/9. Earlier `vitest --runInBand` attempts failed because this Vitest version does not support that flag; reruns used the repo's normal command shape.
- boundary: this is source/dev-app launch detection and settings parity, not a clicked Electron chat transcript, not installed-app parity, not N2 JANG_1L memory clearance, not MiMo JANGTQ2 exactness clearance, not Gemma audio/video E2E, and not release/sign/notarize.
- now: 128GB live proof matrix for the requested Gemma/MiMo/N2 checkpoint lane is captured in `.agents/PROOF_MATRIX_128GB_MIMO_N2_GEMMA_20260610.md`. This is current-source runtime proof, not suite-only bookkeeping.
- Gemma proof: `build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json` is `status=pass` for MXFP4 and JANG4M image/media; `build/current-gemma4-12b-mxfp4-jang4m-live-runtime-audit-20260610.json` is `status=pass` for MXFP4 and JANG4M conservative visible text/multiturn/reasoning-on/required-tool/cache endpoint sanity.
- MiMo proof: `build/current-mimo-v25-jang2l-live-cb-cache-text-20260610.json` is `status=pass`; the 105 GiB JANG_2L bundle loaded on the 128GB host, used `mlx_affine_quantized_matmul`, had about `104997.8 MB` active / `105956.2 MB` peak, preserved exact `ACK-CB-742`, hit paged cache with `cached_tokens=38`, and wrote L2 blocks (`l2_tokens_on_disk=62`). `build/current-mimo-v25-jangtq2-live-cb-cache-text-20260610.json` also loaded/cached, but exact text mutated `ACK-CB-742 -> ACKCB-742`; `build/current-mimo-v25-jangtq2-exactness-variant-probe-live-20260610/result.json` remains `status=open` with all exactness/tool/JSON labels failed.
- N2 proof: `build/current-n2-jangtq2-live-chat-cache-responses-l2-20260610.json` is `status=pass`; the 101 GiB JANGTQ2 bundle loaded on the 128GB host, final memory about `104202.2 MB` active / `105212.9 MB` peak, native cache `hybrid_ssm_v1`, attention TurboQuant KV plus native SSM companion, chat cache `paged+ssm`, required tool, Responses tool/follow-up, Responses stream tool, and fresh-process L2 restore `paged+ssm+disk` all passed.
- N2 red proof: `build/current-n2-jang1l-live-chat-cache-responses-20260610.json` is `status=fail`, `phase=server_startup`; this was launched with `--jang1l-required-extra-headroom-gib 1` and did not preflight-skip. Server log shows qwen3_5_moe/JANG_1L detection, 482 quant-shape patches, 123 shards, bfloat16 for 512 experts, wired limit set to the Metal cap (`115 GB`, model `119 GB` decimal), then abort with `[METAL] Command buffer execution failed: Insufficient Memory`. Boundary: JANG_1L needs a real 128GB loader/runtime memory strategy before release claim.
- now: Gemma QAT/native MXFP4 inventory and objective digest were refreshed against the other-agent Gemma proof artifacts. `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json` remains `status=open`, but all required rows are present and `source_live_smoke_open_rows=[]`; the objective digest records every source-live-smoke artifact while keeping full Gemma release clearance open.
- proof: `build/current-objective-proof-after-n2-jang1l-memory-refresh-20260609.json` now has the Gemma QAT/native row `status=open`, `missing_required_rows=[]`, `source_live_smoke_open_rows=[]`, `all_required_source_live_smokes_present=true`, and `all_required_live_proofs_present=false`. It also now marks `App maxToolIterations cap is enforced for DSV4 tool loop` as `pass` after the refreshed tool-call contract removed stale source-hash drift.
- validation: focused Gemma/objective/tool-call tests passed `17/17`; `py_compile` passed. Boundary: no Gemma release clearance; still need same-model Responses streaming args/content deltas, installed-app/UI settings parity, visual/audio/video honesty per weight-backed modality, media cache salt, mixed-SWA/TQ/L2 proof, and CLI/UI parity.
- now: DSV4 default-cache DSML/tool-loop live gate was retried one-at-a-time for public issue #165/tool-call matrix. It did not load weights; the gate skipped on the existing 120 GiB safety floor with current `psutil.available=112.45 GiB`. The tool-call contract was refreshed afterward and remains `status=open` with no source/panel/parser failures; only `live_default_cache_dsv4_tool_loop_artifact_passed=false`.
- fix: `tests/cross_matrix/run_dsv4_default_cache_tool_loop_gate.py` no longer defaults to the stale missing `panel/release/mac-arm64/.../python3` path. It now resolves Sequoia checkpoint app bundled Python first, then Tahoe, panel bundled Python, `.venv`, and only then the legacy path. Dry-run proof `build/current-dsv4-default-cache-tool-loop-dryrun-current-python-20260609.json` shows the current Sequoia bundled Python and default native prefix/paged/L2 cache flags.
- validation: DSV4 default-cache/tool-loop tests plus tool-call contract open/pass behavior tests passed `14/14`; `py_compile` passed; live gate artifact remains `build/current-dsv4-default-cache-tool-loop/result.json` with `status=skipped`, `reason=insufficient_free_memory`, `required_available_gb=120.0`, `available_gib=112.45`. Boundary: no DSV4 release clearance, no validator weakening, no release/publish action.
- now: Public app issue audit was refreshed after checkpoint packaged integrity. The current pointer is `build/current-public-app-issue-audit-after-checkpoint-packaged-integrity-20260609.json`; it removes stale installed-app hash failures for #111 and #165, and keeps #115/#117/#119/#165 open where live speed, MiniMax reporter parity, Gemma26 memory stress, and full DSML/tool-call contract evidence remain missing.
- proof: regenerated `build/current-release-regression-manifest-after-public-issue-audit-refresh-20260609.json` still has `status=fail`, `prepackage_ready=false`, `release_ready=false`; `public_app_issue_audit=false` only because #165 still has `tool_call_contract_passes=false` from the open tool-call matrix. This is intentional; do not weaken the validator to accept #165 open unless the release scope explicitly defers DSV4/DSML tool-call clearance.
- validation: focused public-audit/current-suite/manifest tests passed `9/9`; `py_compile`, public issue audit runner, release manifest regen, and `git diff --check` passed. Boundary: proof-pointer refresh only, no release/publish action.
- now: Checkpoint packaged-integrity proof is current and green after fixing stale release-gate proof pointers. `panel/scripts/release-gate-python-app.py` now refreshes the current N2/Gemma-aware objective digest, packaged integrity consumes `build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json`, and the release manifest now expects the still-open Gemma QAT/native MXFP4 release requirement instead of treating it as unexpected.
- proof: `build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json` is `status=pass`, `failed=[]`; checks include bundled engine/hash parity, bundled `jang_tools` parity, staged app engine/source parity, no packaged `__pycache__`, hardened runtime/notarization verifier/submit contracts, and `dry_release_gate_uses_current_objective_digest=true`. Regenerated `build/current-release-regression-manifest-after-checkpoint-packaged-integrity-20260609.json` still has `status=fail`, `prepackage_ready=false`, `release_ready=false`, but now reports `component_ok.packaged_integrity_matrix=true`, `installed_app_runtime_parity_audit=true`, and `staged_app_runtime_parity_audit=true`.
- validation: focused tests passed `57/57`; packaged-integrity runner passed; `py_compile` and `git diff --check` passed. Boundary: release remains blocked by real open rows including objective digest/open requirements, tool-call matrix, live smoke/tool smoke, MiMo, real UI matrix, issue175/177/179, public app issue audit, DSV4 memory preflight, plus N2/Gemma media/UI/cache/API runtime proof. No tag, appcast/latest mutation, upload, PyPI publish, or public release was performed.
- now: Checkpoint DMG app runtime parity is current and proof-mapped for both flavors. Ran `run_installed_app_runtime_parity_audit.py` against `panel/release/sequoia-app/mac-arm64/vMLX.app` and `panel/release/tahoe-app/mac-arm64/vMLX.app`; both artifacts are `status=pass`, `missing_or_stale=[]`, with bundled engine hash parity and packaged engine-source hash parity true.
- proof: `build/current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json` and `build/current-installed-app-runtime-parity-audit-tahoe-checkpoint-dmg-20260609.json`. The release manifest now consumes Sequoia as installed-app parity and Tahoe as staged-app parity; regenerated `build/current-release-regression-manifest-after-checkpoint-app-parity-20260609.json` still has `status=fail`, `prepackage_ready=false`, `release_ready=false`, but `current_proof_sweep.component_ok.installed_app_runtime_parity_audit=true` and `staged_app_runtime_parity_audit=true`.
- validation: focused parity/manifest tests passed `23/23`; `py_compile` and `git diff --check` passed. Boundary: this does not clear N2/MiMo/Gemma media/UI/cache live rows, DSV4, MiniMax reporter parity, public tunnel Responses output-index parity, tag/upload/appcast/public release, or PyPI.
- now: Current-source vMLX 1.5.56 checkpoint DMGs are built, Developer ID signed, Apple-notarized, stapled, blockmap-regenerated, and final-verified from this active worktree under Eric's explicit checkpoint override. Artifacts are `panel/release/vMLX-1.5.56-sequoia-arm64.dmg` and `panel/release/vMLX-1.5.56-tahoe-arm64.dmg`.
- proof: build command used the documented keychain path and explicit override: `VMLX_CHECKPOINT_RELEASE_OVERRIDE=1 VMLX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-checkpoint-dmg-override-20260609.json panel/scripts/build-release-dmgs.sh all`. Notarization used `VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh`; final verification used `panel/scripts/verify-release-dmgs.sh`. Both final verifies report `Notarization Ticket=stapled`, `TeamIdentifier=55KGF2S5AY`, `source=Notarized Developer ID`, and `hdiutil` valid checksums.
- hashes: Sequoia SHA-256 `014ef3a9d729bf6b63091e28c82cfe86a9921397aa3d27621cab5f0e0541652f`; Tahoe SHA-256 `272f9c9551fa99332b66c0a686083d94d0b2bf7c5359d310d4983d322dd01686`.
- runtime bundle proof: both Sequoia and Tahoe bundled Python verifiers passed with critical vMLX and `jang_tools` source-content parity, `vmlx_engine 1.5.56`, `mlx 0.31.2`, `mlx-lm 0.31.3`, `mlx-vlm 0.5.0`, Gemma4 unified/VLM/audio register, Qwen3 VL, MiMo V2 register, Step3.7 VLM register, JANGTQ/Kimi VLM loaders, TurboQuant kernels, L2/runtime patches, and Gemma4 pixel-values alias/coercion checks. The app bundle used local `/Users/eric/jang/jang-tools` and installed `jang==2.5.30`; the other-agent `jang==2.5.31` PR is draft/unpublished and was not used.
- boundary: this is a signed/notarized checkpoint DMG build, not a full production-clear release. The override manifest is still `status=fail`, `prepackage_ready=false`, and `release_ready=false`; open rows include tool-call matrix, packaged integrity/current proof sweep, live smoke/tool smoke, MiMo, N2 JANG_1L, DSV4, real UI full matrix, issue179, public app issue audit, and staged/installed runtime parity gaps. No tag, appcast/latest.json mutation, upload, GitHub release publish, PyPI publish, or public download update was performed.
- next other-agent action: if publishing this checkpoint, release notes must call it a checkpoint and list the open rows above. Do not claim N2/MiMo/Gemma/media/tool/UI/cache production clearance from the DMG notarization alone. Rotate the PyPI token pasted in chat before any PyPI action; do not publish `jang==2.5.31` until PR #12 is reviewed/merged and the version is intentionally consumed by vMLX packaging.
- now: Qwen35 same-model direct+panel-gateway raw Responses SSE is live-proven from current source; the remaining failure is isolated to the public tunnel output-index stream. Artifact `build/current-responses-raw-sse-parity-qwen35-direct-gateway-source-vs-tunnel-20260609.json` has all three captures present and still `status=fail` only because tunnel reuses `output_index=0`.
- proof: direct capture `build/responses-sse-captures-20260609/direct-qwen35-mxfp8-mtp-tool-current-source-gateway-run-20260609.sse` and gateway capture `build/responses-sse-captures-20260609/gateway-qwen35-mxfp8-mtp-tool-current-source-20260609.sse` both preserve `record_fact` args `{"value": "blue-cat"}`, include reasoning events, match model `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`, and emit `message=[0]`, `function_call=[1]`. Tunnel preserves args/reasoning/model but emits `message=[0]`, `function_call=[0]`.
- fix/proof harness: added gated panel Vitest live capture `panel/tests/api-gateway-qwen35-live-capture.test.ts`; `tests/cross_matrix/run_qwen35_responses_raw_sse_capture.py` now invokes it while the Qwen35 backend is live. `run_responses_raw_sse_parity_contract.py` now accepts structured gateway logs with `containsReasoning=true` as no-reasoning-disable evidence.
- JANG sync audit: checked `/Users/eric/jang` and `erics-m5-max2.local:~/jang`. Max2 is on `codex/mimo-v25-cache-contract` with dirty JANG converter/package changes in `jang-tools/jang_tools/__main__.py`, `allocate.py`, `capabilities.py`, `convert.py`, and `convert_qwen35_jangtq.py`; these add MLP asymmetry floor control, `processor_config.json` preservation, Qwen35 JANGTQ MTP metadata stamping, and audio/video modality stamps. Do not publish over existing `jang==2.5.30`; if accepted, bump `jang` to `2.5.31`, publish, then bump vMLX dependency/extras and bundled Python.
- next other-agent action: rebuild/redeploy the public tunnel from current source and recapture Qwen35 tunnel raw SSE. Separately, reconcile Max2 JANG dirty files into canonical `/Users/eric/jang/jang-tools`, run jang-tools tests/build/twine check, then only publish a bumped version.
- now: Qwen35 same-model direct/source raw Responses SSE is live-captured against the public-tunnel model id without changing parser/reasoning behavior. Artifact `build/current-responses-raw-sse-parity-qwen35-direct-source-vs-tunnel-20260609.json` is still `status=fail`, but the split is now concrete: current source direct preserves `record_fact` args `{"value": "blue-cat"}`, has reasoning events, reports the same served model as the tunnel, and emits `message=[0]`, `function_call=[1]` with no conflicting output indices.
- proof: direct raw SSE is `build/responses-sse-captures-20260609/direct-qwen35-mxfp8-mtp-tool-current-source-20260609.sse`; server log is `build/responses-sse-captures-20260609/direct-qwen35-mxfp8-mtp-current-source.server.log`. Health shows Qwen35 MXFP8-MTP current-source runtime with native MTP active, `hybrid_ssm_v1` / `hybrid_ssm_typed` cache, TurboQuant attention KV enabled, and block disk cache writes. The tunnel capture still preserves args/reasoning but emits `message=[0]`, `function_call=[0]`; gateway capture is still missing.
- next other-agent action: capture the panel gateway raw SSE with the exact same Qwen35 served model/request. If gateway matches current-source direct (`function_call` at output index `1`), rebuild/redeploy the tunnel backend from current source and recapture tunnel before release. Do not add fake parser fallbacks, disable reasoning, or treat base Qwen/Qwen35-MTP proof as Nex/N2 JANG_1L proof.
- boundary: this narrows Responses #192/#190 failure class but does not clear same-model direct/gateway/tunnel parity, tool-result continuation, UI, installed-app, or release rows. No package/sign/notarize/tag/download action.
- now: N2/JANG_1L no-load memory/index preflight refreshed one-at-a-time before any model launch. Artifact `build/current-n2-pro-jang1l-local-memory-preflight-continuation-20260609.json` reports model exists, index/config/jang_config exist, `artifact_profile=JANG_1L`, `format=jang`, `model_type=qwen3_5_moe`, and `decision=do_not_launch`.
- proof: indexed payload is `110.57 GiB`, required extra Metal/runtime headroom is `8.0 GiB`, required available is `118.57 GiB`, current available is `112.17 GiB`, gap is `6.40 GiB`; weights include `linear_attention_tensors=855`, `vision_tensors=333`, and `expert_tensors=720`.
- boundary: no N2 weights were loaded. This is scheduling/artifact proof only, not runtime/cache/API/UI clearance. Do not lower the gate or force launch below `118.57 GiB` available; retry after freeing RAM or with a real smaller-runtime strategy.
- now: Responses raw-SSE root-cause split rechecked from current source and artifacts. Current source `stream_responses_api()` increments the function-call output item after closing the message item, and the no-heavy source guard still covers empty XML required-arg fail-closed behavior plus output-index ordering. The live Qwen35 public tunnel capture is therefore classified as deployed/tunnel freshness until a same-model current-source direct/gateway capture contradicts it.
- proof: `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-recapture-20260609.json` remains `status=fail` with only the tunnel capture present; the tunnel preserves `record_fact` args `{"value": "blue-cat"}` and has `reasoning_events=10`, but `output_indices_by_type` is `message=[0]`, `function_call=[0]`, so `conflicting_output_indices=[0]`. `build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-e2b-after-parser-20260609.json` remains `status=fail` because direct/gateway Gemma4 E2B preserve args and valid indices while the tunnel returns `model_not_found` for `gemma4-e2b-sse`.
- next other-agent action: do not disable reasoning or add parser fallback injection. Capture Qwen35 current-source direct and panel-gateway raw SSE with the same request/model as the tunnel; if both use output_index `1`, rebuild/redeploy the tunnel backend from current source and recapture tunnel. If direct/gateway also duplicate `0`, then reopen source `stream_responses_api()` before any release claim.
- boundary: no release/package/sign/notarize/tag/download action. Responses parity remains open for same-model direct/gateway/tunnel, actual reasoning events where required, final object consistency, and tool-result continuation on the deployed route.
- now: PyPI checkpoint packages are live. Uploaded `jang==2.5.30` from `/Users/eric/jang/jang-tools` and `vmlx==1.5.56` from this active worktree using fresh `/tmp` build outputs after `twine check` passed for both wheels and sdists.
- proof: PyPI JSON reports `jang 2.5.30` with `jang-2.5.30-py3-none-any.whl` and `jang-2.5.30.tar.gz`, and `vmlx 1.5.56` with `vmlx-1.5.56-py3-none-any.whl` and `vmlx-1.5.56.tar.gz`. A clean temp venv `pip install --no-cache-dir --no-deps jang==2.5.30 vmlx==1.5.56` succeeded and `vmlx` metadata reports `jang>=2.5.30` for hard dep plus `mxtq`/`jang` extras.
- source fix: `pyproject.toml`, `panel/scripts/bundle-python.sh`, `tests/test_engine_audit.py`, and `vmlx_engine/utils/jang_loader.py` now agree on `jang>=2.5.30`. Focused guard `tests/test_engine_audit.py -k "jang_family_runtime or python_dependency_floor or pypi_jang_tools"` passed `2/2`.
- boundary: this is a PyPI package checkpoint, not a signed/notarized DMG release. Full app release remains blocked by the existing prepackage/runtime/model/UI/cache rows.
- now: Gemma 4 12B MXFP4 source image/VL proof is live green at `build/current-gemma4-12b-mxfp4-media-smoke-after-release-doc-correction-20260609.json`. The source server loaded `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4`, advertised vision, processed a red PNG data URL through `/v1/chat/completions`, and returned visible `Red` with `no_channel_leak=true`; health records MXFP4/JANG dispatch and Metal NA active.
- boundary: this is a narrow source API image proof only. It does not clear Gemma tools/cache/UI/installed-app/tunnel/audio/video rows, and no package/sign/notarize/tag/download action was run.
- now: Release/signing docs correction is committed to the active handoff path before any further packaging work. Read `/Users/eric/wiki/infra/apple-notarization.md`; the correct Python/Electron sequence is keychain unlock/partition-list on `~/Library/Keychains/vmlx-build.keychain-db`, then `panel/scripts/build-release-dmgs.sh all`, then `VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh`, then `panel/scripts/verify-release-dmgs.sh`.
- boundary: signing/notary access is known-good; current blocker is `prepackage_ready=false` from the release regression manifest unless Eric explicitly authorizes a checkpoint release with listed open rows. No package/sign/notarize/tag/download action was run in this correction.
- now: Signed checkpoint DMG readiness is explicitly audited and the documented Apple signing path is no longer blocked. Existing local `panel/release/vMLX-1.5.56-sequoia-arm64.dmg` and `panel/release/vMLX-1.5.56-tahoe-arm64.dmg` are Developer ID signed, stapled, and Gatekeeper-accepted, but they are June 5 artifacts and not current-source checkpoint proof for HEAD `d1054f41`.
- proof: `build/current-signed-checkpoint-dmg-readiness-20260609.json` is `status=open`; `existing_dmgs_signed_and_stapled=true`, current installed `/Applications/vMLX.app` is codesign-valid but `Signature=adhoc`, fresh Developer ID signing probe is `pass`, notary history with `--keychain ~/Library/Keychains/vmlx-build.keychain-db --keychain-profile vmlx-notary` is `pass`, and `vmlx-build.keychain-db` reports `no-timeout`.
- build-gate proof: `VMLINUX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-pre-dmg-release-build-after-keychain-unlock-20260609.json panel/scripts/build-release-dmgs.sh sequoia` stopped at the official `--require-prepackage-ready` ledger before building a DMG. The manifest is `status=fail`, `prepackage_ready=false`, `release_ready=false`; `current_proof_sweep.component_ok.packaged_app_developer_id_signing=true`, so this is not a signing/keychain failure.
- N2 refresh: current no-load Nex/N2 Pro 397B JANG_1L preflight still says `do_not_launch`, not because the artifact is missing but because current available headroom is short. `build/current-n2-pro-jang1l-local-memory-preflight-after-release-gate-20260609.json`: payload `110.57 GiB`, required available `118.57 GiB`, observed available `112.56 GiB`, gap `6.01 GiB`. `build/current-n2-jang1l-chat-cache-proof-after-release-gate-20260609.json`: `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`, observed available `112.35 GiB`, gap `6.22 GiB`. No N2 weights were loaded.
- required signing/notary sequence now starts after keychain prep: rebuild current-source Sequoia/Tahoe DMGs, then run `VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh` and `panel/scripts/verify-release-dmgs.sh`.
- boundary: no new signed/notarized current-source DMG was produced in this session; no upload/tag/appcast/public release happened.
- now: Installed-app/package parity slice moved forward without doing a release. `panel/scripts/build-and-install.sh` rebuilt bundled Python and installed `/Applications/vMLX.app` from current source; the app is ad-hoc signed and codesign-valid on disk, not Developer ID signed/notarized. Fixed `panel/scripts/bundle-python.sh` so the Python standalone launcher/runtime files are restored immediately after extraction and again after MLX wheel install, preventing the intermittent missing `python3` / bootstrap encoding failure during rebuild.
- proof: `build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json` is now `status=pass`; bundled imports checked from the installed app include `vmlx_engine 1.5.56`, `mlx 0.31.2`, `mlx-lm 0.31.3`, `mlx-vlm 0.5.0`, TurboQuant disk/cache modules, SSM companion cache/disk store, Gemma4 Unified register, native MTP, and Qwen35 MTP patches. `build/current-packaged-integrity-contract-after-installed-app-rebuild-20260606.json` still fails only on `packaged_app_developer_id_signing_blocked`; release-gate unit contracts pass `47/47` and bundled verifier passes.
- checklist: refreshed `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=73`. Remaining blockers include same-model Responses raw SSE parity/output-index rows, Gemma media/UI/full matrix, MiMo exactness/media/UI, N2 JANG_1L memory-safe live proof, DSV4, MiniMax reporter parity, real UI matrix rows, and real current-source signed/notarized DMG verification.
- boundary: no tag, appcast, notarization, upload, or public release was performed. Do not call the local installed app a signed checkpoint DMG.
- now: MiniMax #179 audit now consumes the current-source MiniMax Small JANGTQ live smoke without closing the reporter K issue. `tests/cross_matrix/run_issue179_minimax_k_root_cause_audit.py` reads `build/current-all-local-model-smoke-minimax-small-jangtq-cache-language-after-bare-invoke-tool-20260609/summary.json` and records concrete green source checks: parser family, tool parser, reasoning parser, reasoning separation, required tool args, tool-result continuation, JSON/code exactness, `paged+tq` second-hit cache, `paged+disk+tq` L2 restart restore, and native TQ/L2 cache capability.
- proof: refreshed `build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json` remains `status=open`; `current_source_minimax_small_smoke.all_checks_pass=true`. The `language_planning_leak_isolation` rows now include source evidence for parser/template/reasoning, paged prefix cache, block-disk L2, and TurboQuant KV, while reporter exact prompt reproduction and reporter generation-config/sampling parity remain open.
- checklist: refreshed `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=73`. New row `issue179_current_source_minimax_small_smoke=true`; red #179 rows remain reporter parity artifact, reporter server hash parity, and current-source cancel probe artifact. Boundary: no release/package/sign/notarize/tag/download, and do not clear MiniMax by disabling cache/TQ/L2/reasoning/sampling or from local-only smoke.
- validation: `tests/test_issue179_minimax_k_root_cause_audit.py` + `tests/test_full_release_objective_checklist.py` passed `37/37`; `py_compile` passed for touched Python files.
- now: Qwen35 MXFP8-MTP startup/health rows are live green without pretending long-tool/cache proof is done. Added `tests/cross_matrix/run_qwen35_mxfp8_mtp_startup.py` and `tests/test_qwen35_mxfp8_mtp_startup.py`; the gate launches `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`, waits for `/health`, records `/v1/cache/stats`, and stops. It does not run Responses turns or long-tool cache rows.
- proof: `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607/00_startup.json` is `status=pass`; `health.model_name=JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`, MTP `native_runtime_active` depth `3`, native cache `hybrid_ssm_v1` / `hybrid_ssm_typed`, TurboQuant attention KV enabled, and routing preserves `trained_active_experts=8`, `effective_active_experts=8`.
- checklist: refreshed `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=84` after Qwen35 startup rows turned green. Remaining Qwen35 blockers are long-tool cache, restart/L2 restore, installed UI/media rows; broader N2/MiMo/Gemma/Responses/package/release blockers remain open.
- validation: `tests/test_qwen35_mxfp8_mtp_startup.py` passed `3/3`, `py_compile` passed, and the live startup gate passed. No signing, notarization, packaging, tag, download, or release action.
- now: Qwen27 JANG_4M-MTP long-context cache/L2 tail proof is live green and checklist-consumed. Added `tests/cross_matrix/run_qwen27_jang4m_mtp_long_context_cache_tail.py` plus focused tests; the harness launches `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP`, runs a 52,035-token cold prompt, stops the server, restarts against the same cache directory, and proves warm restore with visible `LONGCTX-OK`, native MTP D3, hybrid SSM cache, TurboQuant KV, block L2 disk hits, and SSM companion L2 hits. It preserves raw warm stats under `cache_stats_raw` and aggregates cold writes/stores plus warm hits in `phases.warm.cache_stats` for the existing checklist contract.
- proof: `build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json` is `status=pass`; cold prompt tokens/input tokens `52035`, warm cached tokens `52034`, warm cache detail `paged+ssm+disk`, block L2 `disk_writes=814` and `disk_hits=814`, SSM L2 `stores=2`, `hits=1`, `total_tokens_on_disk=104066`, `turboquant_kv_cache.enabled=true`, `mtp.status=native_runtime_active`, `mtp.effective_depth=3`.
- checklist: `tests/cross_matrix/run_full_release_objective_checklist.py` now accepts disk-qualified `paged+ssm+disk` for restart-backed L2 hits instead of requiring only `paged+ssm`. Refreshed `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, but `failed_count` dropped from 101 to 90 and Qwen27 long-context rows are green. Boundary: this is source live proof, not installed-app parity despite the historical artifact filename; Qwen35, N2 JANG_1L, MiMo, Gemma, Responses tunnel/reasoning, installed-app/package/sign/release rows remain open.
- validation: focused long-context/checklist tests passed `5/5`, `py_compile` passed, then the live gate passed. First live attempt exposed a real 413 prompt-too-long overshoot (`~228,049 tokens`); the harness now targets the 30k+ requirement without exceeding the configured 65,536-token cap and does not force `--kv-cache-quantization q4`, preserving native TurboQuant.
- now: Qwen27 MXFP4-MTP API parity is live-proven in source and consumed by the full release checklist. Added `tests/cross_matrix/run_qwen27_mxfp4_mtp_api_parity.py` and `tests/test_qwen27_mxfp4_mtp_api_parity.py`; the harness launches `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` with current `--is-mllm`, exercises Responses text, Responses required `record_fact`, Anthropic Messages, Ollama chat, Chat Completions SSE, prefix cache hit, native MTP health, hybrid SSM/TurboQuant cache health, and a same-cache-dir restart for block/SSM L2 disk hits.
- proof: live run wrote `build/current-qwen27-mxfp4-mtp-api-parity-20260607/summary.json` with `status=pass`; all five API checks returned `ACK`/`record_fact`, `mtp.status=native_runtime_active`, `mtp.effective_depth=2`, `native_cache.schema=hybrid_ssm_v1`, `native_cache.cache_type=hybrid_ssm_typed`, block L2 `disk_writes=5` and `disk_hits=15`, SSM L2 `stores=8` and `total_tokens_on_disk=394`, and scheduler cache hit tokens are positive.
- checklist: regenerated `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`; release remains `status=open`, but `failed_count` dropped from 112 to 101 and the Qwen27 API parity rows are no longer open. Remaining Qwen27 blockers include installed/long-context/UI artifacts, plus Qwen35 rows; this does not clear N2 JANG_1L, MiMo, Gemma, Responses tunnel/reasoning, package parity, signing, or release readiness.
- validation: `tests/test_qwen27_mxfp4_mtp_api_parity.py` passed `4/4`; `py_compile` passed for the new harness/test. First live attempt correctly exposed the stale `--mllm` assumption; harness now uses current CLI `--is-mllm` and has a regression for it.
- now: Gemma4 12B JANG_4M no-media checklist pointer now uses current source live proof instead of a stale missing path. Fresh smoke `build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-current-20260609/summary.json` ran on port 8876 and is `status=fail`, `failed=1`: text cache, multiturn, reasoning, required tool, tool-result continuation, JSON exact, paged+mixed_swa cache hit, and block-disk L2 writes are present, but exact code whitespace emits ` print(add(2, 3))` with a leading space and the mixed-SWA native-cache row reports storage quantization disabled. Refreshed full checklist drops `failed_count` from 119 to 112; remaining Gemma4 12B JANG_4M no-media blockers are exactness and storage-quant policy, not missing evidence.
- now: Responses raw-SSE parity now consumes the local no-heavy Responses streaming contract so #192-style empty XML arguments and output-index regressions are traced separately from deployed same-model tunnel parity. Refreshed `build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-e2b-after-parser-20260609.json` remains `status=fail`, but `local_responses_streaming_guards_pass`, `local_empty_xml_arguments_fail_closed`, `local_output_index_ordering_guard`, and `gateway_argument_stream_passthrough_guard` are all true from `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`. Full checklist remains `status=open`, `failed_count=119`; remaining Responses blockers are same-model tunnel availability/arguments and actual reasoning events, not local `arguments:{}` or duplicate output-index source guards.
- now: Gemma4 12B #191 source startup/visible-generation proof is now consumed by the full release checklist without falsely clearing tools/cache/media/UI. `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=122`, but `gemma4_12b_issue191_startup_artifact_exists`, `gemma4_12b_issue191_startup_status_pass`, and `gemma4_12b_issue191_startup_visible_generation` are green from `build/current-gemma4-12b-issue191-source-startup-visible-proof-20260609.json` (`GEMMA4-OK`, finish `stop`, health checks true). Boundary: old JANG_4M tools/cache nomedia proof and media smoke rows are still missing/red; Gemma full MXFP4/MXFP8/JANG_4M media/cache/API/UI/installed-app/tunnel proof remains open.
- now: N2 JANG_1L one-at-a-time live gate is source/test guarded and currently skipped before launch by real payload/headroom math. User allowed using the 128 GiB machine one at a time, but current available memory is still below the JANG_1L safety gate: `build/current-n2-jang1l-chat-cache-proof-20260609.json` is `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`, `indexed_payload_gib=110.57`, `required_available_gib=118.57`, `available_gib=111.61`, `memory_gap_gib=6.96`. Source fix: `run_n2_chat_cache_gate.py` now applies a JANG_1L-specific indexed-payload preflight even when generic `--min-available-gb` is low, so the proof harness cannot accidentally bypass the post-OOM headroom gate. Objective/checklist now list this live-gate artifact under N2. Boundary: no N2 JANG_1L runtime/API/cache/UI clearance and no release action; next attempt needs enough actual available headroom or a real smaller-runtime strategy, not a lower threshold.
- now: cross-model tool parser package parity coverage is source/test green. Root cause: package/install hash surfaces only pinned `tool_parsers/dsml_tool_parser.py`, while current blockers rely on Qwen, XML-function, Gemma4, MiniMax, auto, and the rest of the registered parser matrix. Added every top-level `vmlx_engine/tool_parsers/*.py` file to bundled-python verifier, release-gate, staged packaged integrity, and installed-app parity hash lists. Red/green proof: four focused package/hash tests failed before wiring on missing parser files, then passed; direct Qwen/Responses empty-XML/output-index behavior slice passed `3/3`; current-suite/release-manifest hash mirror tests passed `4/4`; `bash -n`, `py_compile`, and `git diff --check` passed. Boundary: package/parity guard only, not same-model tunnel raw-SSE, live MiMo/Gemma/N2/UI, or installed-app release clearance.
- now: MiMo current proof pointers are source/test green on the latest cache/exactness evidence. Release/objective/checklist/audit defaults now use `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json` plus `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`; the no-source classifier default input audit was also moved off the older singlebatch speed artifact. Red/green proof: exact-pointer test failed before the update on the stale structured-schema audit pointer, then focused MiMo/release/objective/current-suite tests passed `23/23`; `py_compile` and `git diff --check` passed. Boundary: pointer/gate freshness only; MiMo exactness, strict speed, media/L2/UI/installed-app rows remain open.
- now: Gemma4 vision runtime-bootstrap guard is source/test green. The stale mlxstudio#88 regression checked raw upstream `mlx_vlm.models.gemma4.vision` before importing `vmlx_engine`, so it failed even though current source installs `runtime_patches.gemma4_vision` on engine import. Updated the guard to prove the actual `import vmlx_engine` bootstrap path, including the `_vmlx_gemma4_pixel_values_patch` marker and per-item `mx.array` coercion. Red/green proof: focused test failed before the edit, then passed `1/1`; `py_compile` passed. Boundary: source bootstrap guard only, not Gemma media/UI/installed-app/tunnel release clearance.
- now: runtime patch package parity coverage is source/test green. `vmlx_engine/runtime_patches/__init__.py` auto-installs `deepseek_v4_register`, `gemma4_processing`, `gemma4_vision`, `kimi_k25_mla`, `mlx_lm_compat`, and `mlx_vlm_compat`, but hash gates only covered three of those. Added `runtime_patches/deepseek_v4_register.py`, `runtime_patches/gemma4_vision.py`, and `runtime_patches/kimi_k25_mla.py` to bundled-python/release-gate/packaged-integrity/installed-app parity hash lists; added all auto-installed runtime patch modules plus `tests/test_kimi_k25_mla_patch.py` to current-suite source hashes. Focused tests failed before wiring on missing runtime patch files, then passed `6/6`. Boundary: package/source-hash coverage only, not live Gemma/Kimi/DSV4 release clearance.
- now: Qwen/N2 native-MTP package parity coverage is source/test green. Upstream `ml-explore/mlx-swift-lm` PR #323 reinforces that Qwen3.6/`qwen3_5` depends on hybrid linear-attention/GatedDelta cache correctness. Local root cause: package hash gates covered only `patches/mlx_vlm_mtp/qwen35_vl.py`, not `native_mtp.py` or `patches/mlx_lm_mtp/{__init__.py,batch_generator.py,cache_rollback.py,deepseek_v4_model.py,qwen35_model.py}`. Added all of them to bundled-python/release-gate/packaged-integrity/installed-app parity hash lists. Focused package/parity tests failed before wiring on missing native-MTP files, then passed `4/4`. Boundary: package/parity coverage only, not live N2/Qwen cache/API/UI release clearance.
- now: TQ-native disk cache package parity coverage is source/test green. `vmlx_engine/tq_disk_store.py` is the compressed `TurboQuantKVCache` L2 disk encode/decode path, and `vmlx_engine/cache_record_validator.py` guards TQ-native metadata plus live/prefix/block/SSM/JANGTQ cache restores before unsafe allocations. Both were covered by focused runtime tests but missing from bundled-python/release-gate/packaged-integrity/installed-app parity hash lists. Focused package/parity tests failed before wiring on missing `cache_record_validator.py` / `tq_disk_store.py`, then passed `4/4` after the manifest fix. Boundary: package/parity coverage only, not live N2/MiMo/Gemma cache/UI release clearance.
- now: Qwen/N2 hybrid TurboQuant package parity coverage is source/test green. `vmlx_engine/utils/hybrid_tq_cache.py` controls selective live TurboQuant KV for Qwen3.6/N2 attention layers while preserving SSM companion caches, but was missing from bundled-python/release-gate/packaged-integrity/installed-app parity hash lists. Focused package/parity tests failed before wiring on missing `utils/hybrid_tq_cache.py`, then passed `6/6` after the manifest fix, with `py_compile` and `git diff --check` clean. Boundary: package/parity coverage only, not live N2 cache/tool/UI clearance.
- now: Gemma4 Unified packaged parity hash coverage is source/test green. Source already resolves `mlx_vlm.models.gemma4_unified` after `import vmlx_engine`, and `verify-bundled-python.sh` already import-gates it; the missing piece was installed-app/package parity hash coverage. Added `models/gemma4_unified_register.py` plus `models/gemma4_unified/{__init__.py,config.py,gemma4_unified.py,processing_gemma4_unified.py}` to the release gate, installed-app parity audit, and packaged-integrity hash lists. Red/green focused proof: failed on missing `models/gemma4_unified_register.py`, then passed `4/4`. Boundary: no rebuild, no signing, no installed-app green claim; Gemma media/cache/UI/tunnel rows remain open.
- now: release blocker ledger written to `.agents/RELEASE_BLOCKER_LEDGER_2026_06_09.md`. Active focus is Python vMLX engine + MLXStudio app only, not deprecated `/Users/eric/vmlx`, ADLab, Max2 transport, or old Swift lanes.
- now: single-active cache `max_kv_size` hybrid guard is source/test green. Checked upstream `ml-explore/mlx-lm` PR #1343 and did not blindly backport generic KV rotation for Gemma/MiMo/N2/Qwen hybrid or mixed sliding/full cache paths because it can evict global/full-attention state. `vmlx_engine/utils/single_batch_generator.py` now suppresses generic `max_kv_size` for Gemma4/Gemma4 Unified, MiMo V2, Qwen3.6/N2, explicit hybrid cache, hybrid/mixed/SWA/SSM/Mamba subtypes, or mixed sliding+full/global layer configs, while plain KV models still receive the cap. Focused verification passed `20/20`, plus `py_compile` and `git diff --check`. Boundary: no-heavy cache-policy guard only; live model/UI/release rows remain open.
- current hard boundaries: no fake fixes, no forced sampling/parser/cache/reasoning behavior to hide bugs, no source-vs-quant heavy comparisons if RAM-blocked, no package/sign/notarize/tag/download release until runtime/model/UI/cache/installed-app rows are green or Eric explicitly overrides.
- JANG_1L RAM note: N2/JANG_1L should fit with careful RAM handling; treat it as a careful live-proof scheduling/memory-discipline blocker, not permanent infeasibility. Do not launch below the preflight headroom gate after the Metal OOM evidence.
- active blocker list to keep synchronized: Responses streaming tool args with reasoning enabled and direct/gateway/tunnel parity; MiniMax random Chinese/planning leak isolation; MiMo JANGTQ_2/JANG_2L exactness/cache/tools/media/UI proof; N2/Qwen-family JANG/JANGTQ parser/cache/MTP/gdn_sink proof; DSV4 native SWA/CSA/HCA live proof; Gemma4 QAT/native MXFP4/MXFP8/JANG full media/cache/UI/installed-app/tunnel proof; Step3.7 honest VLM/text boundary plus tool dialect loops; structured JSON/XML repair as hygiene only; UI/CLI/API parity; package/sign/notarize release gate.
- added Responses sub-issue: trace why streaming code near `if tc_args:` sees empty `tc_args` when reasoning is on, including `_parse_tool_calls_with_parser` and streaming delta accumulation. Do not "fix" by disabling reasoning effort.
- added gateway/session issue: Responses failures may be Cloudflare tunnel/gateway/model availability/port/wake/sleep/session routing issues rather than engine parser bugs. Prove local direct vs gateway vs tunnel before classification.
- added Gemma packaged blocker: `ModuleNotFoundError: No module named 'mlx_vlm.models.gemma4_unified'` must be fixed in bundled/installed runtime parity, not only source.
- other-agent integration requirement: before release, ensure all other-agent fixes are included in active source, pushed `origin/main`, bundled Python, installed app, and proof artifacts. Credit GitHub `@Hornsan1` in next release notes/changelog/public acknowledgement.
- now: Gemma4 shared-KV MLX-format load compatibility is source/no-heavy green. `vmlx_engine/runtime_patches/mlx_vlm_compat.py` now drops materialized `k_proj`/`v_proj`/`k_norm`/`v_norm` weights for shared-KV layers in both `sanitize()` and `load_weights()` paths, including `language_model.model.layers.*` and `model.language_model.model.layers.*` key prefixes. Regression coverage added in `tests/test_mlx_lm_runtime_patches.py`; packaged hash-gate coverage in `tests/test_engine_audit.py` now explicitly includes `runtime_patches/__init__.py`, `runtime_patches/mlx_lm_compat.py`, and `runtime_patches/mlx_vlm_compat.py`. Validation passed: focused runtime-patch/hash-gate tests `3/3`, `py_compile`, and `git diff --check`. Boundary: source/no-heavy only; no Gemma QAT live media/cache/UI/installed-app release clearance.
- now: installed-app runtime parity is explicitly red after the current source fixes. `build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json` is `status=open`; `installed_bundled_engine_hash_parity=false` and `installed_packaged_engine_source_hash_parity=false`. Bundled/source mismatches include 15 engine files (`server.py`, `scheduler.py`, `mllm_scheduler.py`, `mllm_batch_generator.py`, `api/tool_calling.py`, `cli.py`, `engine/simple.py`, `models/mllm.py`, `tool_parsers/dsml_tool_parser.py`, `utils/jang_loader.py`, `utils/single_batch_generator.py`, `utils/mlx_vlm_compat.py`, `utils/ssm_companion_cache.py`, `utils/ssm_companion_disk_store.py`, `runtime_patches/__init__.py`) and bundled Python is missing `runtime_patches/mlx_lm_compat.py` plus `runtime_patches/mlx_vlm_compat.py`. `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json` remains `status=pass`, so this is package/runtime staleness, not settings wiring. Do not sign/notarize/tag/release until the app is rebuilt and parity is green.
- now: Step3.7 VLM audit proof refreshed without fake media clearance. Regenerated `build/current-step37-vlm-runtime-audit-after-source-live-media-proof-20260607.json` is `status=pass`, `release_clearance=audit_does_not_block_release`, and `mlx_vlm_step3p7_runtime_available=true`; the audit still records `source_owned_runtime_progress.release_clearance=source_runtime_surface_present_needs_live_proof`, with `live_media_proof.exists=false` and `live_media_proof.pass=false`. Refreshed full checklist `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=121`; stale Step3.7 missing-audit rows are gone, but release remains blocked by Responses tunnel/reasoning, N2, MiMo, Gemma full matrix, Qwen proof gaps, DSV4, and packaging/UI rows.
- now: MiniMax #179 local generation-config fallback is source/test green. The audit now falls back to direct local metadata when `build/current-issue179-minimax-k-local-model-manifest-20260527.json` is absent, hashes only config/generation_config/jang_config/tokenizer/index metadata, and intentionally leaves 67 safetensor shard parity open. Refreshed `build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json` remains `status=open`; `generation_config_and_sampling` is only `partial`, with reporter-machine generation-config hash parity, resolved sampling kwargs parity, raw SSE/session/cancel lifecycle, and full model shard/codebook parity still required.
- now: Responses raw-SSE parity now separates missing reasoning events from a reasoning-disable workaround. Refreshed `build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-e2b-after-parser-20260609.json` is still `status=fail`: direct/gateway preserve `record_fact` args `{"value": "blue-cat"}`, parse cleanly, use valid output indices, match `gemma4-e2b-sse`, and server logs prove `Reasoning: ENABLED` plus `enable_thinking=True`; current tunnel capture is present but returns `model_not_found` for `gemma4-e2b-sse` instead of streaming args. Direct/gateway still emit `reasoning_events=0`. Full checklist `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=121` after the Step3.7 source-runtime audit refresh. Boundary: not release clearance; next proof is same-model tunnel raw SSE plus actual reasoning events without disabling/hiding reasoning.
- live tunnel model-list check: `https://testapi.adlabus.dev/v1/models` currently returns 11 models and does not advertise `gemma4-e2b-sse`; advertised Gemma entries are `Gemma-4-12B-it-MXFP8-CRACK` and `models/Gemma-4-12B-it-MXFP8-CRACK`. This makes the same-model Gemma4 E2B tunnel failure a deployed tunnel/session routing availability blocker, not a local parser-only bug.
- now: MiniMax #179 audit now uses direct local artifact metadata when the old local manifest JSON is absent. Refreshed `build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json` remains `status=open`, but local metadata proves `/Users/eric/models/dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK` has `generation_config.json` (`sha256=2c24fe5507e260bb081727e2a14693d9a982942e721b28720c184d201bffb9dd`), `jang_config.json`, tokenizer/config/index metadata, JANGTQ runtime files, and 67 shards. `generation_config_and_sampling` is now `partial`, not `open`; shard hashes were intentionally not computed by the fallback, and reporter-machine generation config/sampling parity remains open.
- now: MiniMax #179 wrong-language/planning audit has an explicit `language_planning_leak_isolation` matrix in `build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json`. Axes tracked: reporter exact prompt reproduction, generation config/sampling, parser/template/reasoning, paged prefix cache, block-disk L2, and TurboQuant KV. Status remains `open`; the local same-request fallback probe is clean, generation config/sampling is partial, parser/cache/L2/TQ are only partial from local/current-path evidence, and reporter-machine raw SSE/session/cancel lifecycle proof is still required. Boundary: do not clear by disabling cache/TQ/L2/reasoning or changing sampling without same-prompt single-axis A/B proof.
- now: Gemma QAT/native MXFP4 inventory gate separates metadata-advertised media from weight-backed/runtime-proven media and now accepts 12B audio only as honestly gated, not native-audio-proven. Current artifact `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json` remains `status=open`; E2B/E4B have audio tower and vision weights, 12B has image weights but only `embed_audio.embedding_projection.weight` for audio, and 12B/26B/31B video-token metadata still requires live frame-through-vision proof. Full checklist `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` remains `status=open`, `failed_count=121`, with explicit Gemma media rows still open. Boundary: no release/sign/package; Gemma still needs live Responses/tool/media/cache/UI/installed-app/tunnel proof.
- now: N2 JANG_1L careful-RAM boundary tightened and verified. Current artifact `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json` is no-load `status=open`, `decision=do_not_launch`, `indexed_payload_gib=110.57`, `available_gib=114.64`, `required_extra_headroom_gib=8.0`, `required_available_gib=118.57`, `memory_gap_gib=3.93`. JANG_1L should fit just fine as long as RAM is handled carefully and enough headroom is actually available; current local headroom is not enough after the observed Metal OOM. Do not call it impossible, and do not launch below the gate.
- source/proof-harness fix landed in `8caefd24`: `run_n2_chat_cache_gate.py` now emits a structured `status=fail` artifact when the server exits before health or during request probes, preserving OOM/startup evidence instead of dropping the proof. Validation passed: `tests/test_n2_jang1l_memory_preflight.py`, `tests/test_n2_chat_cache_gate.py`, the N2 objective row test, current-suite source-hash test, `py_compile`, and `git diff --check`.
- now: Qwen proof-map audit completed without relabeling narrower proofs. Qwen27 cancel/restart rows have current artifacts, but Qwen27 API parity and long-context tail artifacts are absent in this checkout; Qwen35 startup/long-tool/restart build artifacts are absent while installed UI/video JSON proofs exist. These remain real release-open evidence gaps, not checklist bugs. Tracker stale Qwen27 restart path corrected from `20260607` to `20260609`.
- active release blocker ledger: Responses streaming tool args still need same-model direct/gateway/tunnel raw SSE with reasoning events; MiniMax Chinese/planning leakage still needs cache/parser/L2/generation-config isolation; MiMo still needs JANG_2L/JANGTQ exactness, tools, media, L2 restore, and UI proof; N2 needs memory-safe live JANG_1L/JANGTQ cache/API/UI proof including `gdn_sink`/MTP/hybrid cache/parser rows; DSV4 remains memory-gated; Gemma QAT/native MXFP4 source smokes are green but installed-app/UI/tunnel/full matrix remains open; Step3p7 needs honest scoped VLM/text claims; JSON/XML repair is hygiene, not runtime coherence; no signing/notarization/tag/download until these are green or Eric explicitly overrides.
- now: N2 JANG_1L careful live proof attempted and classified. Conservative launch on port `8899` loaded to server startup, then aborted with Metal OOM: `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)` after `Wired limit set to 115 GB (model 119 GB)`. Source fixes in progress: N2 JANG_1L preflight now requires `8.0 GiB` Metal/runtime headroom (`required_available_gib=118.57`), and `run_n2_chat_cache_gate.py` now returns a structured `status=fail` artifact on early server abort instead of dropping the evidence. Refreshed preflight: `available_gib=114.8`, `memory_gap_gib=3.77`, `decision=do_not_launch`.
- now: pushed `dc6eda78` (`Allow N2 preflight schedule decision`) to `origin/main` and `origin/codex/pr-intake-manifest` on top of other-agent tip `a9cebad3`. Current N2 JANG_1L preflight can now return `schedule_live_proof` when RAM is sufficient; objective test accepts that while keeping N2 release row open until live runtime/cache/API/UI proof exists. Focused validation passed `4/4`, py_compile and diff-check passed.
- now: Responses raw-SSE parity source gate now explicitly tracks `no_reasoning_disable_workaround`. Regenerated `build/current-responses-raw-sse-parity-direct-gateway-gemma4-e2b-after-parser-20260609.json` is `status=fail`: direct/gateway preserve `{"value": "blue-cat"}` args, server logs prove reasoning was enabled, but actual `reasoning_events=0`; tunnel is missing. Full checklist no longer fails `responses_raw_sse_parity_no_reasoning_disable_workaround`, remains `status=open`, `failed_count=121`. Boundary: this is stricter proof classification, not release clearance.
- now: pushed `bb2e4cb7` (`Track Gemma QAT objective blocker`) and follow-up `ac66181f` (`Add N2 JANG1L memory preflight runner`) to `origin/main` and `origin/codex/pr-intake-manifest`. Objective digest now has an explicit Gemma QAT/native MXFP4 E2B/E4B/12B/26B/31B release blocker; source smokes are green but full live proofs remain red. N2 JANG_1L no-heavy careful-RAM preflight runner/tests are source-hashed in current suite. Validation passed: focused tests `6/6` plus follow-up `4/4`, py_compile and diff-check passed.
- now: N2 JANG_1L no-heavy memory/index preflight generator is in progress locally. Refreshed `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json` from the actual local index/config without loading weights: `indexed_payload_gib=110.57`, `required_available_gib=114.57`, `available_gib=113.5`, `memory_gap_gib=1.07`, `decision=do_not_launch`, `classification=careful_ram_live_proof_pending`. Boundary: this is scheduling proof only, not live runtime/cache/API/UI clearance.
- now: pushed `f0386693` (`Clarify N2 JANG1L RAM boundary`) to `origin/main` and `origin/codex/pr-intake-manifest`. Objective digest now frames N2 JANG_1L as careful-RAM live-proof scheduling, not permanent infeasibility; focused N2 objective test passed `1/1`, py_compile and diff-check passed.
- now: pushed `7ef4aa15` (`Update Gemma QAT inventory tracker`) to `origin/main` and `origin/codex/pr-intake-manifest`. Release tracker now points Gemma local QAT/native MXFP4 inventory at `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`, with source-smoke rows green but all five release rows still open for full live/installed-app/UI/tunnel/media proof.
- now: pushed `3bd1d7f4` (`Use unit-labeled MiMo cache proof`) to `origin/main` and `origin/codex/pr-intake-manifest`. MiMo current audit now consumes `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json`; focused tests passed `26/26`, py_compile and diff-check passed. Boundary: pointer sync only, MiMo exactness/media remain open.
- now: pushed `8c1d9222` (`Use Gemma QAT source smoke inventory`) to `origin/main` and `origin/codex/pr-intake-manifest`. Current suite and Gemma QAT inventory default now point at `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`; focused tests passed `5/5`, py_compile and diff-check passed. Boundary: proof-pointer only; full Gemma release proof remains open.
- now: pushed `8141929e` (`Label MiMo cache proof memory units`) to `origin/main` and `origin/codex/pr-intake-manifest`. MiMo cache-vs-no-cache proof harness now labels memory as GiB and treats RAM preflight skip as successful harness exit; focused pytest `5/5`, py_compile, and diff-check passed. Boundary: proof-harness only, MiMo exactness/media remain open.
- now: pushed `66ef200c` (`Use raw SSE checklist in current suite`) to `origin/main` and `origin/codex/pr-intake-manifest`. Current regression suite now runs/allows-open the raw-SSE-aware full objective checklist artifact `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`; focused pointer tests passed `11/11`, py_compile and diff-check passed.
- now: pushed `c27f1024` (`Track Responses raw SSE release parity`) to `origin/main` and `origin/codex/pr-intake-manifest`. Full release objective checklist now includes `responses_raw_sse_parity` for direct/gateway/tunnel raw SSE tool-argument streaming. Refreshed artifact `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` is `status=open`, `failed_count=119`; direct/gateway Gemma4 E2B captures are green but tunnel capture is missing. Boundary: proof-map visibility only, no release/signing/notarization.
- now: N2/JANG_1L RAM boundary clarified by user: it should fit with careful RAM handling. Treat current N2/JANG_1L blocker as careful live-proof scheduling and memory discipline, not permanent infeasibility; no source-vs-quant or extra-heavy comparison unless explicitly allowed.
- now: pushed `7e19117c` (`Track Gemma QAT source smokes and N2 L2 proof`) to `origin/main` and `origin/codex/pr-intake-manifest`. Gemma QAT inventory now records all five required source-smoke proof paths separately from full release live proof; refreshed objective checklist is still `status=open`, `failed_count=116`, with source-smoke row green and release live-proof row red. N2 chat/cache gate now records requested probes and has an explicit L2 restart probe requiring fresh-process block-disk and SSM companion disk hits. Validation passed: py_compile, focused pytest `21/21`, and diff-check. Boundary: no release/signing/notarization; MiMo exactness/media, N2 JANG_1L live proof, Gemma installed-app/UI/Responses matrix, DSV4, MiniMax cache-language isolation, and package gate remain open.
- now: added cross-architecture fix compatibility ledger locally. Every fix must be checked by family, attention/cache architecture, quant format, API surface, UI/CLI launch path, and media capability before any release-wide claim. Important boundaries: hybrid SSM L2 fix is not generic KV proof; Responses args must come from real parser output; Gemma audio/video must be weight-backed; MiMo exactness must not be papered over by parser/JSON/cache/sampling changes; N2 JANGTQ2 proof does not clear JANG_1L; DSV4 requires native SWA/CSA/HCA cache.
- now: Qwen3.6-27B MXFP4 MTP hybrid SSM restart/L2 current-source row is green after SSM disk prefix discovery plus SSM L2 budget policy fix. Live artifact `build/current-all-local-model-smoke-qwen36-27b-mxfp4-mtp-tools-l2-after-ssm-disk-budget-fix-20260609` has `status=pass`, `failures=0`, restart `cached_tokens=56`, `cache_detail=paged+ssm+disk`, block disk `disk_hits=1`, SSM disk `hits=1`, and no `hybrid_kv_without_ssm` fallback.
- validation: `tests/test_n2_chat_cache_gate.py tests/test_ssm_companion_cache.py` passed `61/61`; `py_compile` and `git diff --check` passed for changed source/test files. Boundary: source/live Qwen27 row only; installed app/UI, Qwen35 deployed parity, MiMo exactness/media, Gemma full matrix, DSV4 memory-gated proof, and package/sign/notarize remain open.
- now: pushed `d7fefe7d` LFM2.5-VL projector LayerNorm loading shim to `origin/main`; runtime patch tests passed `10/10`. Boundary: source/no-heavy load compatibility only, not live LFM2.5 VL media/UI release proof.
- now: N2 chat/cache gate memory preflight unit labeling is in progress locally. Artifact `build/current-n2-chat-cache-memory-preflight-after-unit-label-20260609.json` is `status=skipped`, `unit=GiB`, `available_gib=98.08`, `required_available_gib=120.0`, `memory_gap_gib=21.92`; no N2 model launch happened.
- now: pushed `2e989f81` Qwen3-VL/MoE chunked deepstack visual prefill alignment to `origin/main`; runtime patch tests passed `9/9`. Boundary: source compat only, not full VL/UI release proof.
- now: MiMo JANGTQ2 exactness is still red. Current-source tools+L2 smoke `build/current-all-local-model-smoke-mimo-v25-jangtq2-tools-l2-after-responses-n2-20260609` failed 5 exact sentinel rows; conservative no-prefix/no-L2/no-KV-quant isolation `build/current-mimo-v25-jangtq2-nocache-exactness-isolation-20260609.json` still mutates `blue-cat`/`B7-CAT-09`, so cache infrastructure is not the immediate cause.
- now: N2 JANGTQ2 live chat/cache/Responses proof is green: `build/current-n2-jangtq2-chat-cache-responses-proof-after-responses-parser-20260609.json` has `status=pass`; covers chat cache hit `paged+ssm`, Responses required tool/follow-up, streaming tool args, TurboQuant KV recompression, q4 KV storage, paged cache, and block-disk/SSM L2. Installed UI/media/release rows remain open.
- now: Responses raw-SSE parity contract commit `50923a7d` tracks duplicate function/message output-index bugs. Current source passes output-index separation and reasoning-tool-argument rows `2/2`; public tunnel Qwen35 still shows stale/deployed duplicate-index behavior, so same-model deployed parity remains open.
- now: Responses raw SSE parity classifier now also requires same-model direct/gateway/tunnel metadata and valid output item indices by default in the current-suite command. Diagnostic artifact `build/current-responses-raw-sse-parity-mixed-model-tunnel-output-index-20260609.json` is `status=fail`: public tunnel Qwen35 MXFP8 MTP preserves `{"value":"blue-cat"}` args, but it is not the same model as direct/gateway Gemma4 E2B and it reuses `output_index=0` for message and function_call.
- now: Responses finalization source patch prefers real parsed tool calls with non-empty arguments across content/reasoning/full-text candidates; no argument synthesis. Focused server Responses streaming tests passed `2/2`; py_compile and diff-check passed.
- now: Responses public tunnel available-model proof is captured: Qwen35 MXFP8 MTP over `https://testapi.adlabus.dev/v1/responses` streamed `response.function_call_arguments.delta/done` with authoritative args `{"value": "blue-cat"}` and `parse_errors=0`. Same-model Gemma4 E2B parity remains open because the tunnel does not serve `gemma4-e2b-sse`; Gemma4 12B tunnel capture hit `model_load_timeout`.
- now: upstream MLX runtime intake added Gemma4 Unified/shared-KV source shims. `mlx-lm` PR #1349 is locally backported via `gemma4_unified -> gemma4` remapping plus `vision_embedder.*` sanitize filtering; `mlx-vlm` PR #1301 is backported by marking Gemma4 shared-KV layers `kv_shared_only`, removing unused K/V modules, and filtering stale shared K/V weights. Proof: `tests/test_mlx_lm_runtime_patches.py` passed `8/8`; current-suite source-hash mirror slice passed `2/2`; `py_compile` and `git diff --check` passed. Boundary: source/no-heavy only; no Gemma live row, installed app, package, signing, notarization, tag, download, or release action.
- now: MiniMax Small JANGTQ current-source bare-invoke parser fix is live-proven. Artifact `build/current-all-local-model-smoke-minimax-small-jangtq-cache-language-after-bare-invoke-tool-20260609/summary.json` has `status=pass`, `failures=0`; parser subset passed `20/20`. Boundary: reporter parity/random Chinese/visible planning and installed-app parity remain open.
- now: Gemma4 QAT/native MXFP4 E2B/E4B/12B/26B/31B source smoke rows are green after the quoted tool-result harness fix and Gemma4 audio capability gate; other-agent `mlx_vlm` Gemma4 video kwargs compat is inspected and verified with focused tests.
- now: Responses direct local SSE for Gemma4 E2B QAT tool calls is fixed/proven after a Gemma4 parser no-end-marker native-call patch.
- Responses direct proof: `build/current-responses-raw-sse-parity-direct-gemma4-e2b-after-parser-20260609.json`, `status=open` only because `gateway` and `tunnel` captures are missing; direct capture has `argument_delta_count=2`, `argument_done_count=1`, `function_name=record_fact`, authoritative args `{"value": "blue-cat"}`, parse errors `0`.
- Responses gateway proof: `build/current-responses-raw-sse-parity-direct-gateway-gemma4-e2b-after-parser-20260609.json`, `status=open` only because `tunnel` capture is missing; direct and gateway captures both have `argument_delta_count=2`, `argument_done_count=1`, authoritative args `{"value": "blue-cat"}`, expected args match, parse errors `0`.
- Responses root cause: live model emitted complete `<|tool_call>call:record_fact{value:<|"|>blue-cat<|"|>}` without `<tool_call|>`; `Gemma4ToolParser` required the end marker, so streaming buffered heartbeats but never emitted `response.function_call_arguments.*`. Parser now accepts complete closed-brace calls only at end-of-output; partial calls still fail closed.
- source/proof-harness fix: `bench/all_local_model_smoke.py` now quotes the exact tool-result continuation target (`"STORED blue-cat."`) and says the final period is part of the literal. Validation remains strict; this is not a missing-period relaxation.
- live proof: `build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`, `build/current-all-local-model-smoke-gemma4-e4b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`, `build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`, `build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json`, and `build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json` all report `status=pass`/`failed=0`.
- source capability fix: `_bundle_declares_native_audio()` now requires native Gemma4 `audio_config` plus `audio_tower.*` weights before advertising audio; token metadata alone does not schedule audio. 26B/31B now report `text/vision/video` with `audio: not_advertised`, matching zero audio tower weights.
- covered surfaces in those source smokes: text coherency, multi-turn recall, required tool call, tool-result continuation, JSON/code exactness, image/video where emitted, mixed-SWA cache telemetry, and block-disk L2 restart restore; E2B/E4B also clear audio with the `<turn|>` placeholder fix.
- other-agent source included: `vmlx_engine/runtime_patches/mlx_vlm_compat.py` filters unused HF Gemma4 video processor kwargs and is wired into bundled/source hash gates. Focused regression `test_gemma4_video_processor_accepts_hf_config_kwargs` passed.
- still open before release: Responses public tunnel raw SSE parity; MiniMax random-language/cache isolation; MiMo/N2 live parser/cache/tool rows; DSV4 memory-gated live proof; UI/CLI/installed-app parity; package/sign/notarize.
- validation this slice: `py_compile` for changed Python files passed; focused smoke harness tests passed `2/2`; Gemma4 video compat + harness tests passed `3/3`; hash-gate tests passed `3/3`; Gemma4 audio modality tests passed `3/3`; live E2B/E4B/12B/26B/31B QAT smokes passed. No package/sign/notarize/tag/download/release action.

- now: Gemma4 QAT audio turn-marker fix is live-proven for E2B/E4B and ready for scoped commit.
- blocker reduced: Gemma4 QAT/native MXFP4 `media` row. Root cause for E2B/E4B empty audio output was OpenAI `input_audio` placeholder insertion after the Gemma4 `<|turn>model` generation prompt because Gemma4 QAT templates terminate user turns with `<turn|>`, not only `<end_of_turn>`/`<|im_end|>`.
- source fix: `_ensure_gemma4_audio_placeholders()` now recognizes `<turn|>` and inserts missing `<|audio|>` before the user-turn terminator, so the processor expands audio tokens inside the user turn before the model generation prompt.
- live proof: `build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-fullmedia-tools-l2-after-audio-turn-marker-20260609/summary.json` and `build/current-all-local-model-smoke-gemma4-e4b-qat-mxfp4-fullmedia-tools-l2-after-audio-turn-marker-20260609/summary.json` both drop to one remaining failure: exact tool-result punctuation. E2B `audio_blue` returned `Blue`; E4B `audio_blue` returned `blue`; both have `l2_restart_probe` `disk_hits=1`, image/video `Blue`, and post-audio text recovery green.
- 12B QAT proof: `build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-fullmedia-tools-l2-after-audio-turn-marker-20260609/summary.json` also has only the exact tool-result punctuation failure; image/video pass. The runner did not emit an audio row because the runtime capabilities classify this bundle as vision modality, despite suspicious config metadata.
- artifact boundary: E2B/E4B have full audio tower weights; 26B/31B have zero audio weights and must not be claimed audio-capable; 12B has only `embed_audio.embedding_projection.weight` in the index and needs explicit capability honesty before any audio claim.
- validation: new `<turn|>` regression failed before the marker fix and passed after; focused Gemma4 audio tests passed `4/4`; live E2B/E4B/12B QAT smokes above completed. No package/sign/notarize/tag/download/release action.

- now: upstream MLX runtime intake source patch is in progress/verified locally. Added `vmlx_engine/runtime_patches/mlx_lm_compat.py` for confirmed local `mlx_lm` gaps: BatchRotatingKVCache `meta_state` string-bool parsing, LFM2 sigmoid MoE routing, and Gemma4 channel-thinking detection. Preserved the in-flight Gemma4 `<turn|>` audio-placeholder fix.
- proof so far: `tests/test_mlx_lm_runtime_patches.py` passed `4/4`; Gemma4 audio placeholder turn terminator test is included in the next focused run. Runtime patch files/tests are wired into source-hash boundaries and documented in `docs/internal/UPSTREAM_MLX_RUNTIME_INTAKE_2026_06_09.md`.
- other-agent reminder: do not blindly port upstream PRs without local mapping. `<tool_call` marker buffering is already partly covered; missing required XML args still fail closed and must not be synthesized from preamble text. No release/sign/package action.

- now: Gemma4 audio `input_features` forwarding source slice is verified locally and ready to commit with the in-flight MLLM diff.
- blocker reduced: #191/#188 `media` source path after 26B QAT audio failed with `audio_processor_payload_missing`.
- source fix: Gemma4/Gemma4 Unified processor-returned `input_features` and `input_features_mask` are promoted from `extra_kwargs`, salted into media cache keys, and forwarded to the model as native `input_features`/`input_features_mask`; Gemma4 audio prompts missing native audio placeholders append the processor audio token once per audio item; `_run_vision_encoding_inner` uses `getattr` for the optional new fields so older request-like probes do not regress.
- proof: `tests/test_mllm_scheduler_cache.py -k "audio or processor_direct"` plus explicit Gemma4 `input_features`/placeholder tests passed `9/9`; `tests/test_gemma4_audio_waveform_decode.py` passed `1/1`; `py_compile` and `git diff --check` passed.
- #192 user-added issue status: kept on the list as #192/#190 subcase, but do not trust the quoted root-cause wording as current-source truth. Current server tests for `output_index`, required empty XML rejection, and streamed preamble plus empty XML passed `3/3`; current source fails missing required `cmd` closed with `tool_calls_required` and emits no executable `arguments:"{}"` payload for that request shape.
- other-agent reminder: do not synthesize `cmd` from visible preamble text; missing required XML args must fail closed. Still need rebuilt/installed-app and raw direct/gateway/tunnel SSE proof before public closure of #192.
- boundary: source/unit proof only. Gemma4 QAT audio semantic live proof, installed-app/UI parity, full Responses streaming parity, and release readiness remain open. No package/sign/notarize/tag/download/release action.

- now: source fix `e4264d5b` (`Fix Gemma4 sidecar hydration and audio inputs`) is already pushed to `origin/main` and `origin/codex/pr-intake-manifest`.
- Gemma4 QAT/native MXFP4 root cause: 26B-A4B expert weights had 11 cross-shard native-MXFP sidecar splits where `experts.*.weight` and `.scales` were in different safetensor shards; the VLM loader skipped those experts, `strict=False` ignored packed keys, and random `SwitchGLU` expert weights caused multilingual/thought token soup.
- source/proof: VLM loader now hydrates Gemma4 MoE `.scales`/`.biases` from `model.safetensors.index.json` before split/dequant to `experts.switch_glu.*`; Gemma4 audio requests decode temp WAV paths to float32 waveforms before processor call; no-heavy artifact `build/current-model-artifact-format-contract-after-gemma4-cross-shard-sidecars-audio-waveform-20260609.json` is `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest passed=179`.
- live 26B QAT proof: `build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-nomedia-after-cross-shard-expert-sidecars-20260609b/summary.json`; prior incoherence is cleared for text/cache/multiturn/tools/JSON/code/image/video/L2. Remaining failures are exact final period in tool-result continuation and honest unsupported Gemma4 QAT audio feature payload.
- release boundary: no installed-app proof, package, signing, notarization, tag, release, or download update. Gemma4 QAT full matrix remains open for audio semantic bridge, installed-app/UI parity, Responses streaming/live tunnel parity, and the same proof across E2B/E4B/12B/31B rows.
- now: pushed fixes-only commit `0f750579` (`Document XML response format boundary`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- #187 docs boundary fix: `docs/guides/server.md` and `docs/reference/configuration.md` now document `response_format` support for `json_object`, `json_schema`, and `xml`, including `xml_root_tag`, `required_xml_fields`, non-streaming XML-only correction retry, and strict streaming XML `xml_validation_failed` errors.
- validation: docs-boundary regression failed before the docs update on missing XML fields; after the fix the docs test passed, no-heavy API/cache contract passed with `status=pass`, `missing_markers=[]`, `response_format_docs_repair_validation_boundary=true`, `structured_xml_retry_after_repair_failure=true`, and `structured_xml_stream_validation=true`; `py_compile` and `git diff --check` passed.
- release boundary: docs/source only. No packaging, signing, notarization, tag, release, download, installed-app proof, or live-family clearance.
- now: pushed fixes-only commit `64841f44` (`Validate strict XML streaming formats`) to `origin/main` and `origin/codex/pr-intake-manifest`; it follows the other agent's `17377a97` (`Prove N2 Responses tool continuation`).
- #187 XML streaming validation fix: Chat streaming `response_format={"type":"xml", ... , "strict": true}` and Responses streaming `text={"type":"xml", ... , "strict": true}` now validate the final accumulated stream with `repair_xml_output()` and emit XML-specific strict error SSEs (`xml_validation_failed`) when required XML fields/root constraints fail, instead of falling through the JSON validator or no-op path.
- proof-map fix: no-heavy API/cache contract now includes `test_streaming_chat_strict_xml_validates_final_text` and `test_streaming_responses_strict_xml_validates_final_text`, exposes `structured_xml_stream_validation=true`, and release-manifest expected checks include that row.
- validation: the two new streaming XML tests first failed with no error events; after the fix structured XML/JSON retry/guided-hint focus passed `8/8`, current-suite hash/manifest/structured guard tests passed `3/3`, no-heavy API/cache contract passed with `status=pass`, `missing_markers=[]`, `structured_xml_stream_validation=true`; `py_compile` and `git diff --check` passed.
- release boundary: this is stream-end validation only. It does not add streaming XML retry, universal hard grammar-constrained decoding, installed-app proof, packaging, signing, notarization, tag, release, or download readiness.
- now: pushed fixes-only commit `6c815188` (`Hash N2 chat cache gate`) to `origin/main` and `origin/codex/pr-intake-manifest`; it follows `f42ca32e` and the other agent's `0c265976` (`Prove N2 chat tools in cache gate`).
- #190/N2 proof-map fix: current regression-suite source hashes now include `tests/cross_matrix/run_n2_chat_cache_gate.py` and `tests/test_n2_chat_cache_gate.py`, and focused regression pytest now selects `n2_chat_cache_gate`, so the new N2 chat/tool/cache proof harness is not outside the umbrella proof boundary.
- validation: source-hash guard failed before wiring; after the fix `tests/test_n2_chat_cache_gate.py` plus current-suite hash/mirror tests passed `5/5`; `py_compile` and `git diff --check` passed.
- release boundary: proof-map pin only. It does not run/clear N2 JANG_1L live runtime, N2 UI/cache/media, DSV4, installed-app parity, package integrity, signing, or release readiness.
- now: pushed fixes-only commit `f42ca32e` (`Retry strict XML response formats`) to `origin/main` and `origin/codex/pr-intake-manifest`; it was rebased/fast-forwarded on top of the other agent's `0c265976`.
- #187 XML API parity fix: Chat Completions `response_format={"type":"xml", ...}` and Responses `text={"type":"xml", ...}` now use `repair_xml_output()` plus a single XML-only correction retry for strict XML failures. Chat `ResponseFormat` now preserves extension keys like `xml_root_tag`, `required_xml_fields`, and `strict` instead of dropping them during Pydantic coercion.
- proof-map fix: no-heavy API/cache contract now runs the Chat and Responses XML retry tests and exposes `structured_xml_retry_after_repair_failure=true`; release-manifest expected checks include that row; `vmlx_engine/api/models.py` is included in the no-heavy and current-suite source-hash boundaries.
- validation: XML retry tests first failed with the old JSON-only path; after the fix `tests/test_server.py -k "strict_retries_failed_xml_only or strict_retries_failed_json_only or forwards_guided_json_hint"` passed `6/6`; focused no-heavy/current-suite guards passed `2/2`; no-heavy API/cache contract is `status=pass`, `missing_markers=[]`, `structured_xml_retry_after_repair_failure=true`; `py_compile` and `git diff --check` passed.
- release boundary: API/source/no-heavy proof only. This does not claim universal hard grammar-constrained decoding, streaming XML retry, installed-app proof, packaging/signing/notarization, or release readiness.
- now: pushed fixes-only commit `30fa08ee` (`Hash MiMo cache logprob proof`) to `origin/main` and `origin/codex/pr-intake-manifest`; it follows the other agent's `2534bbe0` (`Classify N2 VLM logprob proof boundary`).
- proof-map fix: the current regression-suite source hashes now include the other-agent MiMo cache-vs-no-cache next-token logprob runner and new regression file, and the focused regression command now selects `cache_vs_nocache` tests so the row runs in the umbrella proof.
- validation: new source-hash guard failed before wiring; after the fix `tests/test_mimo_v2_cache_vs_nocache_next_token.py` plus focused current-suite hash/mirror tests passed `5/5`; `py_compile` and `git diff --check` passed.
- release boundary: this pins/source-hashes the other-agent no-heavy/live-proof harness classification only. It does not run/clear live MiMo exactness, DSV4, N2 live runtime/UI/cache/media, installed-app parity, package integrity, signing, or release readiness.
- now: pushed fixes-only commit `08d69a25` (`Label remote DSV4 memory units`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- #190 remote Max2 DSV4 proof-harness fix: `tests/cross_matrix/run_remote_max2_dsv4_readiness.py` now labels its binary `vm_stat` memory gate payload with `unit="GiB"`, `free_plus_speculative_purgeable_gib`, `required_available_gib`, and `memory_gap_gib`, while preserving legacy `*_gb` fields. This aligns the remote Max2 exactness/real-UI guard lane with the local DSV4 tool-loop, route-mode, real-UI, and restart/L2 GiB evidence.
- validation: new regression first failed on missing `memory.unit`, then the remote Max2 guard tests passed `13/13`; focused current-suite source-hash and release-manifest mirror tests passed `2/2`; `py_compile` and `git diff --check` passed.
- release boundary: no package/sign/notarize/tag/download work. This fixes remote DSV4 preflight/proof interpretation only; live DSV4 default-cache tool-loop, live real-UI DSV4, installed-app parity, and release readiness remain open.
- now: pushed fixes-only commit `c2d73ce5` (`Track Responses tool status recovery`) to `origin/main` and `origin/codex/pr-intake-manifest`; it follows the other agent's `773380af` (`Recover Responses tool args from SSE events`).
- #190/#192 proof-map fix: `panel/tests/tool-status-responsiveness.test.ts` now reads panel source relative to the test file so it passes under repo-root proof-runner commands; no-heavy API/cache contract now runs the Responses argument-recovery row via `panel_tool_status_contracts` and exposes `panel_tool_status_responses_argument_recovery=true`.
- validation: root-level tool-status command failed before the test path fix and passed after; focused current-suite/no-heavy gateway guards passed `3/3`; no-heavy contract artifact is `status=pass`, `missing_markers=[]`, `panel_tool_status_contracts passed=1`, `panel_tool_status_responses_argument_recovery=true`; `py_compile` and `git diff --check` passed.
- release boundary: no package/sign/notarize/tag/download work. This does not clear live DSV4 tool-loop, live real-UI DSV4, installed-app parity, or release readiness.
- now: pushed fixes-only commit `5c72b0e8` (`Track Responses gateway proof rows`) to `origin/main` and `origin/codex/pr-intake-manifest`; it was rebased on top of the other agent's `8ff395b7` (`Cover Responses tool buffer cleanup`).
- #190/#192 proof-map fix: no-heavy API/cache contract now runs the other-agent gateway tests `passes Responses function-call argument SSE through unchanged` and `returns backend-unavailable for stale Responses session ports`, and exposes `gateway_responses_function_call_arguments_streaming=true` plus `gateway_stale_responses_port_rejection=true`. Current-suite/release-manifest hashes now include `panel/tests/tool-status-responsiveness.test.ts`.
- validation: proof-map regression failed before wiring and passed after; focused current-suite/no-heavy/release-manifest guards passed `3/3`; refreshed `build/current-noheavy-api-cache-contract-after-structured-schema-decode-20260609.json` is `status=pass`, `missing_markers=[]`, `panel_gateway_contracts passed=4`, both new gateway checks true; `py_compile` and `git diff --check` passed.
- release boundary: no package/sign/notarize/tag/download work. This does not clear live DSV4 tool-loop, live real-UI DSV4, installed-app parity, or release readiness.
- now: pushed fixes-only commit `b114cf54` (`Hash DSV4 restart L2 tests`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- proof-map fix: current regression-suite source hashes now include `tests/test_dsv4_responses_restart_l2_gate.py`; release manifest mirrors the same source hash list, so the DSV4 restart/L2 unit-label regression cannot silently drift out of the proof boundary.
- validation: `tests/test_current_regression_suite.py::test_current_regression_suite_hashes_dsv4_generation_boundary_sources`, `tests/test_current_regression_suite.py::test_current_regression_suite_source_hash_list_matches_release_manifest`, and `tests/test_dsv4_responses_restart_l2_gate.py` passed `4/4`; `py_compile` and `git diff --check` passed.
- now: pushed fixes-only commit `4e62954e` (`Label DSV4 restart L2 memory units`) to `origin/main` and `origin/codex/pr-intake-manifest`; it was rebased on top of the other agent's `c5b30f57` (`Cover stale Responses gateway ports`).
- #190 DSV4 Responses restart/L2 proof-harness fix: `tests/cross_matrix/run_dsv4_responses_restart_l2_gate.py` now labels psutil memory snapshots and memory-skip preflight artifacts with `unit="GiB"`, `available_gib`, `total_gib`, `required_available_gib`, and `memory_gap_gib`, while preserving legacy `*_gb` aliases.
- validation: new restart/L2 unit-label regressions failed before the fix and passed after; `tests/test_dsv4_responses_restart_l2_gate.py` passed `2/2`; `py_compile` and `git diff --check` passed.
- release boundary: this fixes restart/L2 preflight/proof interpretation only. It does not rerun live DSV4 restart/L2, default-cache tool-loop, real-UI DSV4, package integrity, installed-app parity, signing, or release.
- now: pushed fixes-only commit `eb1900ab` (`Label real UI DSV4 memory units`) to `origin/main` and `origin/codex/pr-intake-manifest`; it was rebased on top of the other agent's `690440a2` (`Cover Responses gateway tool argument streaming`).
- #190 real-UI DSV4 proof-harness fix: `tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py` now labels binary memory values with `unit="GiB"` and explicit `*_gib` fields for psutil, required/free/min floor, free+speculative+purgeable gate memory, available-for-gate, and memory gaps, while preserving legacy `*_gb` fields. The focused default-out test now matches the current `build/current-real-ui-dsv4-memory-preflight-after-lfm-step-manifest-fix-20260604.json` pointer used by release-manifest tests.
- validation: new unit-label regressions failed before the fix and passed after; `tests/test_real_ui_dsv4_memory_preflight.py` passed `10/10`; `tests/test_release_regression_manifest.py::test_release_regression_manifest_current_sweep_uses_latest_live_smoke_artifacts` passed; `py_compile` and `git diff --check` passed.
- release boundary: this fixes DSV4 real-UI memory proof interpretation only. It does not run/clear live DSV4 real UI, default-cache tool loop, package integrity, installed-app parity, signing, or release.
- now: pushed fixes-only commit `a5026198` (`Track structured repair score contract`) to `origin/main` and `origin/codex/pr-intake-manifest`; current remote tip also includes the other agent's `39d293fc` (`Cover Responses streaming tool arguments`) on top of my `47f26133`.
- #187 proof-map fix: no-heavy API/cache contract now tracks `test_repair_records_reports_raw_and_repaired_rates`, includes `bench/structured_output_repair_report.py` in source hashes, and release-manifest expected no-heavy checks include `structured_repair_report_rates`.
- validation: contract pin test failed before runner/manifest wiring and passed after; refreshed `build/current-noheavy-api-cache-contract-after-structured-schema-decode-20260609.json` is `status=pass`, `missing_markers=[]`, `checks.structured_repair_report_rates=true`, structured repair command `passed=9`; focused selector passed `4 passed, 137 deselected`; `py_compile` and `git diff --check` passed.
- coordination: other agent's Responses streaming tool-argument tests in `39d293fc` passed locally with `.venv/bin/python -m pytest -q tests/test_server.py -k "streaming and tool"` -> `3 passed, 89 deselected`; no source conflict found.
- now: pushed fixes-only commit `47f26133` (`Report structured repair validation rates`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- #187 benchmark/reporting fix: `bench/structured_output_repair_report.py` now emits explicit score fields in the summary (`raw_json_ok_rate`, `raw_schema_ok_rate`, `raw_xml_ok_rate`, `raw_fields_ok_rate`, `validated_rate`, `invalid_rate`, `repair_needed_rate`) derived from the existing raw/repaired counters. This lets benchmark consumers compare raw parse/schema success against repaired/validated success without recalculating or confusing invalid stored payloads for usable structured output.
- validation: new regression failed before the fix with missing `raw_json_ok_rate`; after the fix `tests/test_structured_output.py tests/test_structured_output_repair_report.py` passed `56 passed, 2 skipped`; `py_compile` and `git diff --check` passed.
- release boundary: fixes-only. No package/sign/notarize/tag/download action was run. #187 remains open for broader live/benchmark adoption, streaming limitations, XML API parity, installed-app proof, and any future universal hard grammar-constrained decoding claims.
- now: pushed fixes-only commit `e97608f4` (`Label DSV4 route memory units`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- source/proof-harness fix: `tests/cross_matrix/run_dsv4_route_mode_code_exactness.py` now labels psutil/vm-stat memory snapshots and preflight artifacts with explicit binary-unit fields (`unit="GiB"`, `available_gib`, `total_gib`, `required_available_gib`, `available_for_gate_gib`, `memory_gap_gib`) while preserving legacy `*_gb` fields for existing consumers.
- validation: new route-mode unit-label regression failed before the fix; after the fix `tests/test_dsv4_route_mode_code_exactness.py` passed `27/27`; `py_compile` and `git diff --check` passed.
- release boundary: user explicitly said no releases. No package/sign/notarize/tag/download action was run.
- now: pushed fixes-only commit `2f395057` (`Avoid storing invalid structured payloads`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- #187 source fix: `bench/structured_output_repair_report.py` no longer writes schema-invalid JSON/XML into `structured_output.parsed` / `structured_output.xml` as if it were usable structured data. Invalid rows keep `is_valid=false`, error, raw parse/schema diagnostics, and repair actions, but the structured payload field is `None`.
- root cause: `repair_json_output()` keeps parsed invalid objects for diagnostics, and the benchmark JSONL writer passed that through. A schema-invalid row like `{"visible_text": 123}` could be stored as `{"visible_text": [123]}` under `parsed` even though `raw_schema_ok=false`.
- validation: new regression failed before the fix, then `tests/test_structured_output.py tests/test_structured_output_repair_report.py` passed `55 passed, 2 skipped`; `py_compile` and `git diff --check` passed.
- now: pushed fixes-only commit `c04ac543` (`Label DSV4 gate memory units`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- source/proof-harness fix: `tests/cross_matrix/run_dsv4_default_cache_tool_loop_gate.py` now labels psutil memory snapshots as binary units with `unit="GiB"`, `available_gib`, and `total_gib`, while preserving legacy `available_gb`/`total_gb` fields for existing manifest consumers. Memory skip artifacts also include `required_available_gib` and `min_free_gib`.
- root cause: the prior artifact showed `total_gb=128.0` and `available_gb=113.59` from division by `1024**3`; a separate decimal GB sample could show about `122.0 GB`, causing confusion even though the gate was correctly below the `120 GiB` launch floor.
- validation: new regression failed before the fix, then `tests/test_dsv4_default_cache_tool_loop_gate.py` passed `11/11`; `py_compile` passed for changed files; `git diff --check` passed.
- release boundary: user explicitly said no releases. No package/sign/notarize/tag/download action was run in this slice.
- now: pushed `fb50eed0` (`Accept current objective digest in package gate`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- source fix: `tests/cross_matrix/run_packaged_integrity_contract.py` now derives the dry release-gate expected objective-digest failure line from `build/current-objective-proof-after-pr-intake-matrix-refresh-20260609.json`, excluding suite-deferred rows, instead of only the stale short `EXPECTED_OPEN_REQUIREMENTS` list. If the current artifact is absent, it falls back to the static known-open list.
- regression proof: added `tests/test_packaged_integrity_contract.py::test_packaged_integrity_accepts_current_objective_digest_only_failure`; verified red first, then green.
- validation: `tests/test_packaged_integrity_contract.py` passed `50/50`; `py_compile` passed for changed files; `git diff --check` passed; full packaged-integrity runner refreshed `build/current-packaged-integrity-contract-after-bundled-python-sync-20260608.json`.
- packaged-integrity result after fix: `status=fail`, `failed=["packaged_app_developer_id_signing_blocked"]`, `dry_release_gate_fails_only_on_known_objectives=true`, `dry_release_gate_uses_current_objective_digest=true`; bundled verifier checks are true. Remaining package boundaries are stale staged app runtime/source hash parity and Developer ID keychain user interaction (`errSecInternalComponent`) blocking signing use from Codex.
- DSV4 boundary unchanged: latest `build/current-dsv4-default-cache-tool-loop/result.json` is skipped for insufficient memory, required `120.0 GB`, observed about `115.53 GB` psutil available. Do not bypass the memory gate.
- now: pushed combined tip `7e59e38b` to `origin/main` and `origin/codex/pr-intake-manifest`. It merges the other agent's `74d338b4` (`Repair schema-keyed DSML parameters`) with my `43b1cba7` (`Refresh PR intake regression suite pointer`).
- source/test refresh: current regression suite now defaults to `build/current-regression-suite-after-pr-intake-matrix-refresh-20260609.json`; release manifest `CURRENT_REGRESSION_SUITE_ARTIFACT` consumes that same PR-intake suite artifact.
- regenerated evidence: `build/current-regression-suite-after-pr-intake-matrix-refresh-20260609.json` is `status=open`, `failed_steps=["packaged_integrity_contracts", "release_regression_manifest", "release_gate_skip_app"]`; manifest now embeds this artifact and remains `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; objective/checklist remain open.
- validation after merging other agent edit: `tests/test_dsml_tool_parser.py` plus PR-intake pointer/objective/checklist tests passed `138/138`; focused pointer/objective/checklist tests passed `121/121`; `py_compile` passed; `git diff --check` passed.
- GitHub: #190 updated with the DSML + pointer refresh at https://github.com/jjang-ai/vmlx/issues/190#issuecomment-4657711574.
- boundary: DSML schema-key repair is source/test green, but #190 remains open until live DSV4 default-cache tool-loop proof runs above the memory gate. No release/sign/notarize/download claim.
- now: pushed PR-intake proof-pointer refresh `3f8b5c4d` (`Refresh PR intake proof pointers`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- source/test refresh: generation defaults, objective digest, release manifest, full release checklist, and release-gate objective proof defaults now point at the PR-intake matrix artifact set, including `build/current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json`, `build/current-objective-proof-after-pr-intake-matrix-refresh-20260609.json`, `build/current-release-regression-manifest-after-pr-intake-matrix-refresh-20260609.json`, and `build/current-full-release-objective-checklist-after-pr-intake-matrix-refresh-20260609.json`.
- proof state: generation-defaults contract is `status=pass`; release manifest remains `status=fail`, `prepackage_ready=false`, `release_ready=false`; full checklist remains `status=open`, `failed_count=128`.
- validation: focused pointer/gate tests passed `126/126`; `py_compile` passed for changed Python files; `git diff --check` and staged `git diff --cached --check` passed before commit.
- boundary: this is release proof-pointer freshness only. It does not clear live DSV4, MiMo, N2, installed-app parity, packaged integrity, signing, notarization, or public download rows.
- handoff: `.agents/PR_INTAKE_FIX_LEDGER_20260609.md` now lists #186-#192 fixes/proofs and tells other agents to include `3f8b5c4d`, not deprecated `/Users/eric/vmlx`, and not close #190/#191/#192 from public-user perspective without the remaining live/package proofs.
- now: pushed #188 matrix refresh `798c84a6` (`Refresh cross-model metadata route audit`) to `origin/main` and `origin/codex/pr-intake-manifest`.
- proof: `build/current-local-generation-metadata-route-audit-after-pr-intake-20260609.json`, `status=pass`, `rows=15`, `hard_failures=[]`; default high-risk audit now includes local Step3p7 repro paths `dealignai/Step-3.7-Flash-JANG_2L-CRACK`, `dealignai/Step-3.7-Flash-JANG_K-CRACK`, and `JANGQ/Step-3.7-Flash-JANG_K`.
- #188/#189 boundary: Step3p7 advertised-media rows now record `step3p7_source_vlm_runtime_available=true` and note `step3p7_advertised_media_routes_source_vlm_runtime`, so the matrix no longer collapses current source VLM runtime routing into the old unsafe metadata route. This is no-heavy route evidence, not full live family release proof.
- cross-agent ledger: `.agents/PR_INTAKE_FIX_LEDGER_20260609.md` lists #186-#192 fixes, proofs, issue boundaries, and what the other agent must remember before packaging/release.
- now: #191 Gemma4 12B source startup/generation proof is green on the merged source tip `03aa3c16`; no source change was needed in this slice.
- live proof: `build/current-gemma4-12b-issue191-source-startup-visible-proof-20260609.json`, `status=pass`; local `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` resolved both `mlx_vlm.speculative.drafters.gemma4_unified_assistant` and `mlx_vlm.models.gemma4_unified_assistant`, served healthy on port 8874, returned HTTP 200 with visible `GEMMA4-OK`, `completion_tokens=10`, `finish_reason=stop`, and stayed healthy after the request.
- validation: focused Gemma alias tests passed `2/2`; `bash -n panel/scripts/verify-bundled-python.sh` passed; `git diff --check` passed; port 8874 was clear after shutdown.
- boundary: #191 is source-live proven for this local JANG_4M 12B bundle and current alias/startup path; public installed/downloaded apps remain stale until rebuilt/notarized, and MXFP4/MXFP8, audio/video, installed-app parity, and full Gemma matrix remain open.
- now: pushed merge tip `03aa3c16` to `origin/codex/pr-intake-manifest` and `origin/main`, containing `4194892c` (`Repair Qwen plain-line tool calls`) plus the other agent's `e7d0e65f` MiMo classifier commit.
- blocker reduced: #192 `parser/template` for Qwen3.6/Qwen3-Coder tool calls. Live Qwen3.6 35B emitted `bash\necho "Tool test successful!"`; Qwen parser now converts this plain tool-name + body dialect to schema-valid JSON only when the first line exactly matches a request-declared tool with one required string parameter.
- live proof: `build/current-qwen36-35b-issue192-exact-reporter-tool-proof-after-plain-line-fix-20260609.json`, `status=pass`; exact reporter payload returns HTTP 200, `finish_reason=tool_calls`, tool `bash`, arguments `{"command":"echo \"Tool test successful!\""}`, no visible content, and health recovers after request.
- contract proof: `build/current-tool-call-contract-after-cross-model-loop-metrics-20260609.json`, `status=open`, `failed=[]`, `missing_markers=[]`, `qwen_issue_192_plain_tool_line_repaired=true`; open only on `live_default_cache_dsv4_tool_loop`.
- boundary: the same live Qwen server still rejects a stricter `tool_choice=required` second probe when the model emits `<tool_call><function=bash></function></tool_call>` with no `command`; server correctly refuses to invent a missing required argument. #190 DSV4 remains memory-gated.
- GitHub updates: #192 plain-line live proof at https://github.com/jjang-ai/vmlx/issues/192#issuecomment-4657244829; #190 cross-reference at https://github.com/jjang-ai/vmlx/issues/190#issuecomment-4657246578.
- preflight artifact: `build/current-dsv4-default-cache-tool-loop-preflight-refresh-20260609.json`, `status=skipped`, `reason=insufficient_free_memory`, required `120.0 GB`, observed available `111.48 GB` at `2026-06-09T00:06:54-0700`.
- release action: none; no DSV4 server/model launch happened, and no signing/notarization/tag/release/download update is allowed from this state.
- previous source: `ed9af3b4` remains included for #190 required-tool bare JSON argument repair, with #187 live LFM structured response_format proof in the same live run.
- source change: `_parse_tool_calls_with_parser()` now repairs schema-valid bare JSON argument objects only for `tool_choice=required` single-tool requests whose emitted keys/required fields match the exposed tool schema.
- live proof before/after: `build/current-all-local-model-smoke-lfm25-mxfp8-response-format-schema-20260609/.../result.json` had `tool_required` HTTP 400 with `raw_preview={"value":"blue-cat"}`; `build/current-all-local-model-smoke-lfm25-mxfp8-response-format-schema-after-bare-json-tool-fix-20260609/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json` has HTTP 200 parsed `record_fact({"value":"blue-cat"})` and `validation_failures=[]`.
- #187 live proof: same after-fix LFM artifact sends `response_format={"type":"json_schema"}` for `structured_json_exact`, returns exact `{"status":"ok","value":"blue-cat","count":3}`, and has no structured validation failures.
- GitHub #187 updated at https://github.com/jjang-ai/vmlx/issues/187#issuecomment-4657076982; GitHub #190 fix proof updated at https://github.com/jjang-ai/vmlx/issues/190#issuecomment-4657080345, DSV4 preflight boundary updated at https://github.com/jjang-ai/vmlx/issues/190#issuecomment-4657134726, and Qwen plain-line cross-reference updated at https://github.com/jjang-ai/vmlx/issues/190#issuecomment-4657246578.
- regenerated contracts: no-heavy API/cache `status=pass`, `missing_markers=[]`, `structured_live_smoke_response_format_adoption=true`; tool-call contract `status=open`, `failed=[]`, `missing_markers=[]`, `required_single_tool_bare_json_arguments_repaired=true`, `qwen_issue_192_plain_tool_line_repaired=true`, open only on `live_default_cache_dsv4_tool_loop`.
- validation: post-merge focused #192/tool-contract pytest passed `14/14`; `py_compile` passed for Qwen parser, tool-call contract, tests, and merged MiMo classifier; `git diff --check` passed. Earlier `tests/test_tool_format.py` passed `114/114`; no-heavy contract regenerated pass; current-suite + release-manifest passed `401/401`.
- blocked: #190 still remains open for live DSV4 default-cache tool-loop proof; current local memory is below the gate's `120.0 GB` floor. #187 still needs fresh live/benchmark adoption across more families. LFM MXFP8 still fails the independent exact-code whitespace row by missing final `)`. Broader release blockers remain open.
- dirty not to commit: `.agents/LOG.md` and `.agents/STATUS.md` local-only coordination; pre-existing/generated `build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json`, `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`, and ignored tracker `docs/internal/VMLX_MLXSTUDIO_RELEASE_EXECUTION_TRACKER_2026_06_07.md`.
- branch: codex/pr-intake-manifest
- cwd: /Users/eric/mlx/vllm-mlx-finite-launch-guard
- last_update: 2026-06-09 00:22 PDT

## CODEX - 2026-06-09 MiMo token-trace exactness classification
- blocker reduced: `MiMo V2.5 JANGTQ_2` literal exactness / cache-storage classification.
- proof artifact: `build/current-mimo-v2-jangtq2-cb-cache-lossless-token-trace-live-20260609.json`.
- server log: `build/current-mimo-v2-jangtq2-cb-cache-lossless-token-trace-live-20260609.server.log`.
- result: live run `status=pass`; native MiMo mixed-SWA cache remains enabled with prefix=true, paged=true, block_disk_l2=true, storage_quantization.enabled=false; warm request hit `cached_tokens=46`, `cache_detail=paged`; L2 has `78` tokens; no think tags; generation TPS about `59.4` on final health.
- exactness boundary: both cold and warm rows returned visible `ACKCB-742` instead of expected `ACK-CB-742`, so exactness remains red.
- token-trace evidence: cold request selected token sequence `ACK`, `CB`, `-`, `7`, `4`, `2`, `<|im_end|>`; warm cached request selected the same sequence. The hyphen token is generated, so the remaining exactness failure is runtime text assembly / token decoding boundary, not prefix cache, paged cache, L2, stored KV quantization, sampler, or parser mutation.
- release status: MiMo remains release-open until token assembly exactness, required/auto tool rows, tool-result continuation, JANG_2L live rows, media rows, and L2 fresh-process restore are cleared.

## CODEX - 2026-06-09 correction to MiMo token-trace classification
- Correction to the immediately previous MiMo token-trace note: the generated token stream was `ACK`, `CB`, `-`, `7`, `4`, `2`, `<|im_end|>`, so the runtime did not drop a generated hyphen between `ACK` and `CB`; that first delimiter was not generated.
- Updated classification: cache/storage remains cleared for this exactness row because cold and warm runs match and warm cache telemetry is correct. Text assembly is not proven as the cause. The remaining exactness issue is model/template/artifact/runtime-logit behavior before choosing the `CB` token over a delimiter-plus-C/B path.

## CODEX - 2026-06-09 MiMo no-source exactness classifier refresh
- blocker reduced: `MiMo V2.5 JANGTQ_2` exactness classification without source-vs-quant load.
- source fix: `tests/cross_matrix/run_mimo_v2_no_source_exactness_classifier.py` now treats missing stale smoke as empty evidence and accepts both old label-keyed `cases`/`requests` artifacts and current list-shaped live probe artifacts.
- focused verification: `py_compile` passed for classifier + current audit; `tests/test_mimo_v2_no_source_exactness_classifier.py` passed `13/13`.
- exactness artifact: `build/current-mimo-v25-jangtq2-exactness-variant-probe-live-after-lossless-token-trace-20260609/result.json`.
- classifier artifact: `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`, `status=open`, classification `jangtq2_plain_literal_copy_fails_before_parser_or_json_repair`.
- current audit artifact: `build/current-mimo-v2-jang2l-current-audit-after-lossless-token-trace-classifier-20260609.json`, `status=open`.
- evidence: `/v1/completions` mutates `blue-cat -> blue cat` and `B7-CAT-09 -> B7 CAT-09`; chat mutates `blue-cat -> blue grass` and `B7-CAT-09 -> B7 CAT-09`; JSON mutates values to `blue` and `B7CAT-09`; parsed tool calls preserve protocol but mutate arguments to `blue` and `B7CAT-09`.
- classification boundary: this is before JSON repair and before parser rewrite. Tool parser works for this run; argument exactness is wrong. Source-vs-quant remains user-disallowed due RAM, so do not launch source comparison locally.
- release status: not release-ready; do not sign/notarize/tag. MiMo still needs exactness/root-cause fix or model rebuild contract, JANG_2L live rows, required/auto tool continuation, media E2E, L2 fresh-process restore, and installed-app UI proof.
- N2 status unchanged in this continuation: N2 JANGTQ2 narrow proof remains previously green; N2 JANG_1L local launch remains memory-blocked and must not be loaded on this host without enough headroom.

## CODEX - 2026-06-09 N2 runtime/cache/parser status refresh
- blocker reduced: `N2` source/API/cache/family detection status without unsafe local model load.
- no-heavy API/cache contract: `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`; covers Chat/Responses sampling kwargs, max-output/max-context separation, JSON schema format preservation, streaming cache detail, Responses previous_response_id, cache stats/reuse endpoints, TurboQuant KV runtime contract, TurboQuant disk roundtrip, and named cache/parser rows.
- cache architecture contract: `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`; includes `mimo_v2_asymmetric_swa_kv_status`, hybrid SSM companion L2 contracts, TurboQuant disk roundtrip, panel cache launch policy, and family cache matrix completion.
- family detection contract: `build/current-model-family-detection-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`; includes CLI parser choices, panel session launch parser/modality policy, JANG/JANGTQ/MXFP speed row distinction, and high-risk local row detector policy.
- N2 JANG_1L preflight: `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`; decision `do_not_launch`; payload `118.73GB`, host `128GB`, free+speculative `94.26GiB`, required extra headroom `20GiB`.
- release boundary: N2 plumbing/source contracts are green; N2 JANG_1L live runtime remains release-open until tested on a host with sufficient headroom or a smaller runtime strategy. Do not launch it locally in this memory state.

## CODEX - 2026-06-09 N2 explicit family/panel detection row
- blocker reduced: `N2` autodetect/startup/parser/cache policy coverage.
- source changes: added explicit N2 Pro Qwen3.5-MoE hybrid/VL metadata coverage to Python registry tests, panel detector tests, and the no-heavy model-family detection contract.
- current honest policy: N2-shaped affine JANG Qwen-MoE keeps `family=qwen3_5_moe`, hybrid cache, `tool_parser=qwen`, `reasoning_parser=qwen3`, thinking support, and paged-cache requirement; without indexed MTP/VL-ready tensors, it routes text-only (`is_mllm=false` / panel `forceTextOnly=true`) until the real VL path is implemented and live-proven.
- proof artifact: `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`, `status=pass`, `missing_rows=[]`, `n2_pro_qwen35_moe_hybrid_vl_policy=true`; engine/panel/launch commands returned 0.
- focused checks passed: Python N2 registry test, panel N2 detector test, family row pin tests, and py_compile for changed Python files.
- release boundary: this closes an N2 autodetect/panel-policy coverage gap only. It does not clear N2 JANG_1L live runtime, N2 VL/audio/video, cache L2 restart, UI installed-app, or memory-blocked model-load rows.

## CODEX - 2026-06-09 gateway stale-port and standby wake no-heavy contract
- blocker reduced: `api/ui` gateway startup and sleep/wake routing.
- source fix: `panel/src/main/api-gateway.ts` now treats only active local sessions (`running`, `loading`, `standby`) as gateway-port conflicts. Stopped/error local rows and remote sessions can retain stale DB ports after restart/sleep and no longer block gateway startup.
- test added: `panel/tests/api-gateway-single-model.behavior.test.ts` -> `allows gateway startup on ports used only by stopped or remote saved sessions`.
- existing standby wake gateway test remains green: `auto-switches to a standby model by waking it before direct OpenAI streaming`.
- release gate update: `tests/cross_matrix/run_noheavy_api_cache_contract.py` now includes `panel_gateway_contracts` and checks `gateway_stale_port_startup` plus `gateway_standby_wake_routing`.
- proof artifact: `build/current-noheavy-api-cache-contract-after-gateway-stale-port-20260609.json`, `status=pass`, `missing_markers=[]`, `gateway_stale_port_startup=true`, `gateway_standby_wake_routing=true`, `panel_gateway_contracts rc=0 passed=2`.
- boundary: no-heavy app/API proof only. This does not clear installed-app parity, live model routing, media, L2 restart, or release signing.

## CODEX - 2026-06-09 MiMo local structural proof refresh
- blocker reduced: `MiMo V2.5` local manifest/structural proof integrity for JANGTQ_2 and JANG_2L without model load or source-vs-quant comparison.
- source change: `tests/cross_matrix/run_mimo_v2_local_bundle_metadata_contract.py` now emits the local JANGTQ2 manifest and the MiMo structural proof artifact in addition to the metadata contract.
- proof artifacts:
  - `build/current-mimo-v2-local-bundle-metadata-contract-20260607.json`, `status=pass`.
  - `build/current-mimo-jangtq2-local-manifest-20260607.tsv`, verified by current audit with `rows=117`, `ok_files=117`, `total_bytes=85304837837`.
  - `build/current-mimo-jang2l-local-structural-verify-20260606.json`, `status=pass`; covers both local bundles, model-owned `generation_config.json`, `xml_function` tool parser metadata, `think_xml` reasoning parser metadata, hybrid full/SWA cache topology, prefix cache, L2 disk cache, TurboQuant-KV boundary, bookend affine sidecars, stacked `switch_mlp` layout, and no legacy `.mlp.experts.*` layout.
  - `build/current-mimo-v2-jang2l-current-audit-after-mimo-n2-runtime-refresh-20260609.json`, `status=open`, with `manifest_integrity=true` and `structural_verify=true`.
- focused validation: `tests/test_mimo_v2_local_bundle_metadata_contract.py`, `tests/test_mimo_v2_current_audit.py`, `tests/test_mimo_v2_no_source_exactness_classifier.py`, `tests/test_objective_proof_digest.py`, `tests/test_full_release_objective_checklist.py`, and `tests/test_release_regression_manifest.py` passed `466/466`.
- remaining MiMo blockers are real live rows: text-cache proof, SwitchGLU parity proof, cache-vs-no-cache next-token proof, tool protocol, decode speed, source-vs-quant or accepted no-source equivalent, MLLM inputs-embeds proof, block-disk L2 restart restore, image/video live E2E, audio waveform live E2E, JANGTQ2 media/L2, and JANG_2L media/L2.
- release boundary: not release-ready; do not sign/notarize/tag/release.

## CODEX - 2026-06-09 no-heavy objective gate refresh after MiMo structural proof
- blocker reduced: no-heavy `api/ui`, `cache/storage`, `parser/template`, generation-default, native-MTP, and VL media contract freshness.
- refreshed artifacts:
  - `build/current-max-output-context-contract-after-jangtq2-objective-refresh-20260607.json`, `status=pass`; objective row now PASS.
  - `build/current-generation-defaults-contract-after-dsv4-preflight-refresh-20260608.json`, `status=pass`.
  - `build/current-native-mtp-contract-after-noheavy-contract-refresh-20260608.json`, `status=pass`.
  - `build/current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json`, `status=pass`.
  - `build/current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json`, `status=pass`; objective row now PASS.
  - `build/current-parser-registry-contract-after-jangtq2-objective-refresh-20260607.json`, `status=pass`.
  - `build/current-model-artifact-format-contract-after-mllm-tight-memory-guard-20260607.json`, `status=pass`.
  - `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`, `status=pass`; objective high-risk parser/artifact/launch policy row now PASS.
  - `build/current-objective-proof-after-mimo-n2-gateway-pointer-refresh-20260609.json` now shows PASS for max-output/context, cross-family cache architecture, high-risk model-family parser/artifact/launch policy, generation defaults/native-MTP/VL media, and current-source API/non-DSV4 cache contracts.
- remaining objective rows are live-heavy quality/speed/UI/model rows: DSV4 cache/tool/live rows, Qwen/JANG speed/MTP rows, Ling/Gemma quality rows, cross-family live smoke, MiMo live runtime rows, N2 live runtime/UI/cache/media, MiniMax reporter parity, real Electron UI matrices, and DSV4 long-output/code/file quality.
- release boundary: still not release-ready; signing/notarization/download updates remain locked.

## CODEX - 2026-06-09 LFM25 MXFP4 live smoke refresh
- blocker reduced: cross-family live smoke matrix / LFM25 MXFP4 runtime evidence.
- live command: bundled Python all-local smoke, single `LFM2.5-8B-A1B-MXFP4`, tools enabled, no media/video, port 8866.
- proof artifact: `build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json`, `status=probe_failed` with one remaining validation failure.
- passes in artifact: visible `ACK` cache repeat, multi-turn recall (`blue cat`), required `record_fact({"value":"blue-cat"})` tool call, tool-result continuation (`STORED blue-cat.`), structured JSON exact, `tool_parser=lfm2`, `reasoning_parser=qwen3`, typed native cache `hybrid_ssm_v1`, prefix/paged cache, `paged+ssm` and `paged+ssm+disk` cache details, SSM L2 persisted/hit evidence.
- remaining LFM25 MXFP4 failures: exact-code whitespace missing final `)`, block-L2 write/hit checklist still not green from this artifact; LFM25 MXFP8 artifact is still missing.
- harness fix: `bench/all_local_model_smoke.py` now validates tool-result continuation against the actual prompt text `STORED blue-cat.` instead of falsely requiring no period.
- focused validation: `tests/test_all_local_model_smoke.py` and `tests/test_full_release_objective_checklist.py` passed `75/75`.
- release boundary: cross-family live matrix remains open; this converts LFM MXFP4 from missing evidence to current partial live evidence only.

## CODEX - 2026-06-09 LFM25 MXFP8 live smoke refresh
- blocker reduced: cross-family live smoke matrix / LFM25 MXFP8 runtime evidence.
- live command: bundled Python all-local smoke, single `LFM2.5-8B-A1B-MXFP8`, tools enabled, no media/video, port 8867.
- proof artifact: `build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json`, `status=probe_failed` with one remaining validation failure.
- passes in artifact: visible `ACK` cache repeat, multi-turn recall (`blue cat`), required `record_fact({"value":"blue-cat"})` tool call, tool-result continuation (`STORED blue-cat.`), structured JSON exact, `tool_parser=lfm2`, `reasoning_parser=qwen3`, typed native cache `hybrid_ssm_v1`, and `paged+ssm` cache hit telemetry.
- remaining LFM25 MXFP8 failure: exact-code whitespace missing final `)`.
- full checklist artifact: `build/current-full-release-objective-checklist-after-lfm25-mxfp8-live-smoke-20260609.json`, `status=open`, `failed_count=132`; LFM artifacts now present/current rather than missing.
- focused validation: `tests/test_full_release_objective_checklist.py` passed `3/3`.
- release boundary: LFM25 remains open for exact-code quality and MXFP4 block-L2 write/hit criterion; cross-family live matrix remains open.

## CODEX - 2026-06-09 MiMo/N2 runtime-cache-parser pointer refresh
- blocker reduced: `MiMo/N2` release-gate freshness for runtime/cache/parser/TurboQuant evidence without unsafe local heavy loads.
- regenerated proof: `build/current-mimo-v2-local-bundle-metadata-contract-20260607.json` `status=pass`, `build/current-mimo-jang2l-local-structural-verify-20260606.json` `status=pass`, and `build/current-mimo-jangtq2-local-manifest-20260607.tsv`.
- current MiMo audit: `build/current-mimo-v2-jang2l-current-audit-after-mimo-n2-runtime-refresh-20260609.json`, `status=open`; manifest integrity and structural verify are true; prefix/paged/L2 lossless-cache evidence remains present.
- remaining MiMo blockers: text-cache narrow proof, SwitchGLU selected-expert parity, cache-vs-no-cache next-token match, tool protocol/exact args, decode speed, source-vs-quant or accepted no-source equivalent, MLLM inputs-embeds interface proof, block-disk L2 restart restore, image/video E2E, audio waveform E2E, and per-bundle media/L2 rows.
- N2 state: no-heavy API/cache/family-detection contracts remain pass; N2 JANG_1L live launch remains memory-blocked by `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`; current observed local free+speculative memory was about 50.5 GiB.
- source changed this continuation: `tests/cross_matrix/run_mimo_v2_jang2l_current_audit.py`, `tests/cross_matrix/run_mimo_v2_no_source_exactness_classifier.py`, `tests/cross_matrix/run_full_release_objective_checklist.py`, `tests/cross_matrix/summarize_objective_proof.py`, `tests/cross_matrix/release_regression_manifest.py`, pointer tests, and the release tracker.
- release boundary: not release-ready; do not sign/notarize/tag/release from no-heavy pointer freshness.

## CODEX - 2026-06-09 N2 objective/checklist evidence correction
- blocker reduced: N2 release board accuracy for runtime/cache/parser/UI-startup evidence.
- source fix: `tests/cross_matrix/summarize_objective_proof.py` no longer claims the N2 artifact is absent; it consumes the current N2 memory preflight plus no-heavy API/cache/cache-architecture/model-family contracts.
- proof artifacts regenerated: `build/current-objective-proof-after-mimo-n2-runtime-refresh-20260609.json` and `build/current-full-release-objective-checklist-after-mimo-n2-runtime-refresh-20260609.json`.
- current N2 detail: local JANG_1L artifact/index present, model path `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANG_1L`, payload `118.73 GB`, memory decision `do_not_launch`, launch_safe=false.
- no-heavy N2 detail: API/cache, cache architecture, model-family detection, N2 family policy, TurboQuant runtime contract, TurboQuant disk roundtrip, and hybrid cache policy are reflected as pass.
- release boundary: N2 remains open for memory-safe live runtime/API/UI/cache/media proof; no signing/notarization/tag/release.
- focused verification: `py_compile` passed; `tests/test_objective_proof_digest.py::test_objective_proof_digest_tracks_n2_pro_397b_release_blocker` and `tests/test_full_release_objective_checklist.py::test_full_release_objective_checklist_tracks_open_n2_pro_objective_row` passed.

## CODEX - 2026-06-09 MiMo JANGTQ2 live runtime proof refresh
- blocker reduced: MiMo JANGTQ2 live runtime/cache/tool-protocol/speed proof is current instead of inferred from older structural/no-heavy rows.
- live proof artifact: `build/current-all-local-model-smoke-mimo-v25-jangtq2-live-runtime-proof-20260609/JANGQ_MiMo-V2.5-JANGTQ_2/result.json`, `status=probe_failed` with 5 exactness failures.
- cleared by this proof: model loads and serves, visible cache repeat `ACK`, paged prefix-cache hit with `cached_tokens=60`, multi-turn recall `blue cat`, exact code whitespace row, real OpenAI tool-call structure, MiMo media runtime capability reporting, and decode speed `51.45 tok/s`.
- still failing: tool args mutate literal values (`blue-cat` -> `blue-123`, `B7-CAT-09` -> `B7CAT-09`), tool-result continuation omits final period, structured JSON mutates literal values (`blue-cat` -> `blue-1`, `B7-CAT-09` -> `B7CCAT-09`).
- regenerated audit: `build/current-mimo-v2-jang2l-current-audit-after-live-mimo-jangtq2-runtime-proof-20260609.json`, `status=open`; `tool_protocol=true`, `decode_speed_target=true`, `artifact_exactness=false`.
- regenerated objective digest: `build/current-objective-proof-after-live-mimo-jangtq2-runtime-proof-20260609.json`, still open.
- regenerated release checklist: `build/current-full-release-objective-checklist-after-live-mimo-jangtq2-runtime-proof-20260609.json`, `status=open`, `failed_count=129`.
- remaining release blockers: MiMo exactness, MiMo block-disk L2 restart restore, MiMo image/video live E2E, MiMo audio waveform live E2E, JANG_2L live media/L2 proof, and N2 memory-safe live runtime/API/UI/cache/media proof.
- release boundary: do not sign, notarize, tag, or publish downloads from this state.

## CODEX - 2026-06-09 MiMo explicit text-runtime media capability fix
- blocker reduced: MiMo V2.5 JANGTQ_2 no longer advertises fake image/video/audio runtime support when artifact metadata explicitly says `weights_preserved_text_runtime`.
- source fix: `_mimo_v2_media_runtime_auto_enabled()` now refuses to auto-enable media when `capabilities.multimodal_status` or `runtime.multimodal_mode` is explicitly text-only/unwired/preserved-disabled; `model_capabilities()` also avoids generic MLLM vision fallback when MiMo runtime modalities are known.
- live proof: `build/current-mimo-v25-jangtq2-capabilities-textonly-after-explicit-text-runtime-fix-20260609/capabilities.json` and forced-MLLM proof `build/current-mimo-v25-jangtq2-capabilities-forced-mllm-after-explicit-text-runtime-fix-20260609/capabilities.json` both report `modalities=["text"]`, media `runtime_modalities=["text"]`, and vision/image/video/audio as `preserved_unwired`.
- current-source MiMo smoke: `build/current-all-local-model-smoke-mimo-v25-jangtq2-current-source-textonly-l2-after-capability-fix-20260609/JANGQ_MiMo-V2.5-JANGTQ_2/result.json` drops media failures and proves block-disk L2 restart restore (`disk_hits=2`, `cached_tokens=60`), but still fails 5 exactness rows.
- current-source speed probe: `build/current-mimo-v25-jangtq2-current-source-long-decode-speed-after-capability-fix-20260609/result.json` is `36.6 tok/s` wall (`server: 36.9 tok/s`); `max_num_seqs=2` was slower at `32.4 tok/s`. Speed remains below the 40 tok/s target.
- remaining MiMo blockers: exact tool/JSON literal mutations, current-source speed below target, source-vs-quant/logit diagnosis still not done because RAM was disallowed, and true media E2E remains intentionally unavailable for this text-runtime artifact.
- release boundary: do not sign/notarize/tag/release from this state.

## CODEX - 2026-06-09 MiMo exactness runtime-boundary isolation
- blocker investigated: MiMo JANGTQ2 exact literal/tool-arg/JSON mutations.
- fast-path isolation: added `VMLINUX_DISABLE_MIMO_V2_COMPILED_ROUTER=1` and `VMLINUX_DISABLE_MIMO_V2_SWITCHGLU_FAST_PATH=1` diagnostic gates; reran exactness smoke at `build/current-all-local-model-smoke-mimo-v25-jangtq2-disable-router-switchglu-fastpath-exactness-20260609/JANGQ_MiMo-V2.5-JANGTQ_2/result.json`.
- result: exactness still failed the same 5 rows with both vMLX MiMo compiled router and SwitchGLU fast path disabled. This rules out those vMLX fast paths as the primary cause.
- TQ kernel parity: one-layer MiMo JANGTQ2 packed/norm tensors were compared against manual unpack/codebook/norm/inverse-Hadamard reference without full model/source load. Gate/up and sliced down-proj matched within float16 tolerance (`gate/up max_abs_diff=0.0009765625`, `down first256 max_abs_diff=4.77e-7`). This does not support a gathered TQ Metal kernel bug.
- current classification: exactness remains `jangtq2_plain_literal_copy_fails_before_parser_or_json_repair`; strongest current boundary is artifact/logit quality from the 2-bit routed JANGTQ2 profile, not parser/cache/L2/JSON repair/router/SwitchGLU/TQ gather kernel.
- MiMo JANG_2L preflight: `build/current-mimo-v25-jang2l-local-memory-preflight-20260609.json`, payload `112.06 GB`, current free+speculative `75.55 GiB`, decision `do_not_launch`; no JANG_2L load attempted.
- N2 status unchanged: `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json` remains `do_not_launch` for local JANG_1L payload `118.73 GB`.
- release boundary: do not release; next real MiMo progress requires a better artifact/profile or an authorized stronger logits/source comparison on adequate memory.

## CODEX - 2026-06-09 MiMo JANGTQ2 L2 proof pointer correction
- blocker reduced: MiMo JANGTQ2 block-disk L2 restart restore was stale in the current audit/checklist.
- source fix: current MiMo audit now consumes `build/current-all-local-model-smoke-mimo-v25-jangtq2-current-source-textonly-l2-after-capability-fix-20260609/JANGQ_MiMo-V2.5-JANGTQ_2/result.json` instead of the older live-runtime proof artifact for JANGTQ2 all-local smoke evidence.
- proof artifacts regenerated: `build/current-mimo-v2-jang2l-current-audit-after-current-jangtq2-l2-proof-20260609.json`, `build/current-objective-proof-after-current-jangtq2-l2-proof-20260609.json`, and `build/current-full-release-objective-checklist-after-current-jangtq2-l2-proof-20260609.json`.
- result: MiMo audit now has `block_disk_l2_restart_restore=true`, `mimo_jangtq2_l2_restart_cache_hit=true`, and `mimo_jangtq2_l2_restart_visible_output=true`; checklist remains `status=open`, `failed_count=130`.
- remaining MiMo blockers: artifact/literal exactness, decode speed below strict 40 tok/s, source-vs-quant or equivalent artifact diagnosis, MLLM inputs-embeds proof, VL/audio/video unwired, JANG_2L live media/L2, and broader live UI/app release rows.
- focused validation: py_compile passed; `tests/test_mimo_v2_current_audit.py tests/test_full_release_objective_checklist.py tests/test_objective_proof_digest.py` passed `130/130`.
- release boundary: do not sign/notarize/tag/release from this state.

## CODEX - 2026-06-09 MiMo MLLM inputs-embeds interface proof
- blocker reduced: `mimo_mllm_inputs_embeds_interface_missing_or_failed`.
- runtime fix: `vmlx_engine/models/mllm.py` now preserves explicit empty `unwired_modalities` and treats explicit `multimodal_status=media_runtime_enabled` as wired by default, while `weights_preserved_text_runtime` remains text-only/unwired.
- proof runner added: `tests/cross_matrix/run_mimo_v2_mllm_inputs_embeds_interface.py`.
- proof artifact: `build/current-mimo-v2-mllm-inputs-embeds-interface-proof-20260609.json`, `status=pass`; checks direct `inputs_embeds` forwarding plus image/video/audio supplied-embedding splice and text-token preservation.
- audit artifact: `build/current-mimo-v2-jang2l-current-audit-after-mllm-inputs-embeds-proof-20260609.json`, `status=open`; `component_ok.mllm_inputs_embeds_interface=true`; blocker `mimo_mllm_inputs_embeds_interface_missing_or_failed` removed.
- objective/checklist artifacts: `build/current-objective-proof-after-mllm-inputs-embeds-proof-20260609.json`; `build/current-full-release-objective-checklist-after-mllm-inputs-embeds-proof-20260609.json`, still `status=open`, `failed_count=130`.
- focused validation: py_compile passed; proof runner passed; `tests/test_mimo_v2_media_runtime.py tests/test_mimo_v2_mllm_runtime_registration.py tests/test_mimo_v2_current_audit.py tests/test_full_release_objective_checklist.py tests/test_objective_proof_digest.py` passed `147/147`.
- remaining MiMo blockers: text-cache narrow proof, SwitchGLU selected-expert parity, cache-vs-no-cache next-token proof, JANGTQ2 literal/tool/JSON exactness, decode speed below strict 40 tok/s, source-vs-quant or equivalent artifact diagnosis, VL/audio/video live E2E, JANGTQ2 live media/L2 row, and JANG_2L live media/L2 row.
- release boundary: do not sign/notarize/tag/release from this state.

## CODEX - 2026-06-09 MiMo current text-cache proof
- blocker reduced: `mimo_text_cache_narrow_proof_missing_or_failed`.
- source fix: current MiMo audit now accepts narrow text-cache proof from the current all-local JANGTQ2 smoke when `text_cache_repeat_1` and `text_cache_repeat_2` both produce visible exact `ACK` and the second request reports cached tokens.
- proof artifact: `build/current-mimo-v2-jang2l-current-audit-after-current-text-cache-proof-20260609.json`, `status=open`; `component_ok.text_cache_narrow=true`; blocker `mimo_text_cache_narrow_proof_missing_or_failed` removed.
- objective/checklist artifacts: `build/current-objective-proof-after-current-text-cache-proof-20260609.json`; `build/current-full-release-objective-checklist-after-current-text-cache-proof-20260609.json`, still `status=open`, `failed_count=130`.
- focused validation: py_compile passed; `tests/test_mimo_v2_current_audit.py tests/test_full_release_objective_checklist.py tests/test_objective_proof_digest.py` passed `131/131`.
- remaining MiMo blockers: SwitchGLU selected-expert parity, cache-vs-no-cache next-token match, JANGTQ2 literal/tool/JSON exactness, decode speed below strict 40 tok/s, source-vs-quant or equivalent artifact diagnosis, VL/audio/video live E2E, JANGTQ2 live media/L2, and JANG_2L live media/L2.
- release boundary: do not sign/notarize/tag/release from this state.

## CODEX - 2026-06-09 MiMo SwitchGLU selected-expert parity proof
- blocker reduced: `mimo_switchglu_selected_expert_parity_missing_or_failed`.
- proof runner added: `tests/cross_matrix/run_mimo_v2_switchglu_parity.py`.
- proof artifact: `build/current-mimo-v2-switchglu-selected-expert-parity-20260609.json`, `status=pass`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `fast_calls=1`, no fallback reasons.
- audit artifact: `build/current-mimo-v2-jang2l-current-audit-after-switchglu-parity-proof-20260609.json`, `status=open`; `component_ok.switchglu_selected_expert_parity=true`; blocker removed.
- objective/checklist artifacts: `build/current-objective-proof-after-switchglu-parity-proof-20260609.json`; `build/current-full-release-objective-checklist-after-switchglu-parity-proof-20260609.json`, still `status=open`, `failed_count=130`.
- stale test correction: `tests/test_engine_audit.py` now requires MiMo `weights_preserved_text_runtime` metadata with unwired media sidecars to stay text-only even with `force_mllm=True`; this matches runtime guard behavior and avoids fake media routing.
- focused validation: py_compile passed; SwitchGLU proof runner passed; affected engine tests passed `2/2`; audit/checklist/objective tests passed `132/132`.
- remaining MiMo blockers: cache-vs-no-cache next-token match, JANGTQ2 literal/tool/JSON exactness, decode speed below strict 40 tok/s, source-vs-quant or equivalent artifact diagnosis, VL/audio/video live E2E, JANGTQ2 live media/L2, and JANG_2L live media/L2.
- release boundary: do not sign/notarize/tag/release from this state.

## CODEX - 2026-06-09 MiMo artifact-only exactness diagnosis
- blocker reduced: `mimo_source_vs_quant_first_divergence_missing_or_failed`.
- source fix: no-source exactness classifier now records plain literal failures with no cache reuse plus current all-local runtime/native KV quantization disabled; this can satisfy the audit's source-vs-quant policy-skip gate when source-vs-quant is disallowed by RAM.
- classifier artifact: `build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json`, `status=open`; `excluded_surfaces.prefix_paged_l2_or_kv_quant_primary_cause=true`; `model_upload_action_required=true`.
- audit artifact: `build/current-mimo-v2-jang2l-current-audit-after-artifact-diagnosis-20260609.json`, `status=open`; `source_vs_quant_policy_skipped=true`; `source_vs_quant_requirement_satisfied=true`; source-vs-quant blocker removed.
- objective/checklist artifacts: `build/current-objective-proof-after-artifact-diagnosis-20260609.json`; `build/current-full-release-objective-checklist-after-artifact-diagnosis-20260609.json`, still `status=open`, `failed_count=128`.
- focused validation: py_compile passed; `tests/test_mimo_v2_no_source_exactness_classifier.py tests/test_mimo_v2_current_audit.py tests/test_full_release_objective_checklist.py tests/test_objective_proof_digest.py` passed `147/147`.
- remaining MiMo blockers: cache-vs-no-cache next-token match, JANGTQ2 literal/tool/JSON exactness with model upload action required, decode speed below strict 40 tok/s, VL/audio/video live E2E, JANGTQ2 live media/L2, and JANG_2L live media/L2.
- release boundary: do not sign/notarize/tag/release from this state.

## CODEX - 2026-06-09 MiMo cache-vs-no-cache logprobs proof
- blocker investigated: `mimo_cache_vs_nocache_next_token_missing_or_mismatch` / `cache-storage`.
- source added: `tests/cross_matrix/run_mimo_v2_cache_vs_nocache_next_token.py`, a memory-preflighted live MiMo JANGTQ2 one-token logprobs proof with `skip_prefix_cache=true` vs cache warm/hit.
- superseded release-path artifact: `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-20260609.json` is superseded by `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json`, `status=pass`.
- isolation artifact: `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-memory-20260609.json`, `status=fail`.
- current result: paged cache/no-cache proof passes. No-cache, warm-store, and cache-hit rows all emit `ACK` with identical top10/logprob signature; cache hit reports `cached_tokens=31`, `cache_detail=paged`, and telemetry is now explicitly labeled `unit=GiB`. The older memory-aware isolation artifact remains a separate failed diagnostic.
- current audit: `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, `status=open`; `component_ok.cache_vs_nocache_next_token=false`.
- objective/checklist: `build/current-objective-proof-after-cache-vs-nocache-logprobs-20260609.json`; `build/current-full-release-objective-checklist-after-cache-vs-nocache-logprobs-20260609.json`, `status=open`, `failed_count=128`.
- release boundary: do not sign/notarize/tag/release. MiMo cache is not strict lossless at logprob-distribution level; exactness, speed, media, JANG_2L, UI, and N2 live rows remain open.

## CODEX - 2026-06-09 #187 structured XML repair diagnostics
- blocker reduced: #187 XML repair/validation parity for benchmark/catalog callers.
- commit pushed: `8ab0f5da` to `origin/codex/pr-intake-manifest` and `origin/main`.
- source fix: `repair_xml_output()` now reports post-generation XML extraction/validation diagnostics: canonical XML, `raw_xml_ok`, required-field validity, repair actions, and fail-closed ambiguous candidate handling.
- benchmark adapter: `bench/structured_output_repair_report.py` now supports `--format xml`, `--xml-root-tag`, and repeated `--required-xml-field` for JSONL repair summaries.
- no-heavy proof: `build/current-noheavy-api-cache-contract-after-structured-schema-decode-20260609.json` regenerated `status=pass`, `missing_markers=[]`, with `structured_xml_repair_validation_boundary=true`.
- verification: py_compile passed; `git diff --check` passed; structured-output tests passed `53 passed, 2 skipped`; current-suite/release-manifest tests passed `399/399`; no-heavy API/cache contract passed.
- GitHub: #187 updated at https://github.com/jjang-ai/vmlx/issues/187#issuecomment-4656754462.
- boundary: this is post-generation XML repair/reporting only, not hard grammar/guided decoding and not Chat/Responses XML `response_format` retry. #187 still needs guided-decoding investigation/exposure if runtime can enforce it, plus live/benchmark runner adoption against real model outputs.
- release boundary: do not sign/notarize/tag/update downloads from this state.

## CODEX - 2026-06-09 Responses server SSE tool proof-map refresh
- blocker reduced: `api/ui` + `parser/template` proof tracking for Responses streaming tool-call arguments and output item ordering.
- source/proof-map fix: `tests/cross_matrix/run_noheavy_api_cache_contract.py` now runs a dedicated `responses_streaming_tool_contracts` command for server-side Responses SSE tool-call behavior.
- pinned tests: `test_streaming_responses_tool_call_arguments_survive_buffering`, `test_streaming_responses_reasoning_tool_call_keeps_arguments`, `test_streaming_responses_tool_call_uses_next_output_index_without_text`, and `test_streaming_responses_required_empty_xml_tool_call_is_rejected`.
- proof artifact: `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`, `status=pass`, `missing_markers=[]`, `responses_streaming_tool_call_arguments_and_indexes=true`, `responses_streaming_tool_contracts rc=0 passed=4`.
- validation: proof-map test failed before wiring and passed after; current-suite server/gateway/tool-status no-heavy slice passed `6/6`; release-manifest API-surface/current-suite slice passed `3/3`; server Responses streaming tool slice passed `4/4`; py_compile and `git diff --check` passed.
- boundary: this is current-source no-heavy server proof coverage. It does not clear local-vs-tunnel raw SSE comparison, MLXStudio installed-app execution, N2 JANG_1L memory-gated live path, DSV4, MiniMax, MiMo, Gemma media/cache/UI, or release readiness.

## CODEX - 2026-06-09 Responses preamble empty-XML tool-call boundary
- blocker reduced: #192/#190 `parser/template` + `api/ui` proof tracking for the new Qwen/Qwen-Coder report where a visible preamble precedes `<tool_call><function=exec_command></function></tool_call>`.
- root-cause boundary: do not trust/report that current source emits executable `{}` for this shape. Current XML-function parsing can produce `{}` internally, but `_filter_to_request_tools()` drops the call because required `cmd` is missing, and streaming Responses emits `tool_calls_required`.
- source/proof-map fix: added `test_streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments` and wired it into `responses_streaming_tool_contracts`.
- proof artifact refreshed: `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`, `status=pass`, `missing_markers=[]`, `responses_streaming_tool_call_arguments_and_indexes=true`, `responses_streaming_tool_contracts rc=0 passed=5`.
- focused validation so far: new current-suite marker guard failed before wiring and passed after; server Responses streaming tool slice with 5 tests passed.
- boundary: missing required XML args must fail closed; do not invent `cmd` from preamble text. Public #192 remains open until rebuilt/installed app proof; #190 remains open for live DSV4/default-cache/tool-loop and cross-family release rows.

## CODEX - 2026-06-09 Responses gateway reasoning empty-final-args boundary
- blocker reduced: #190/#192 local MLXStudio gateway/panel Responses SSE argument preservation when reasoning is present and final `response.output_item.done.item.arguments` is empty.
- source/proof-map fix: added panel regression `passes Responses argument SSE with reasoning and empty final item arguments`; no-heavy API/cache contract now requires `gateway_responses_reasoning_empty_final_arguments_streaming`, and release manifest expects that check.
- proof artifact: `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`, `status=pass`, `missing_markers=[]`, `panel_gateway_contracts rc=0 passed=5`, `responses_streaming_tool_contracts rc=0 passed=5`.
- other-agent reminder: keep the server fail-closed rule for missing required XML args; do not synthesize tool args from preamble text. This gateway row proves local pass-through/recovery only, not public tunnel parity, rebuilt installed-app behavior, or release readiness.

## CODEX - 2026-06-09 Responses raw SSE parity capture harness
- blocker reduced: #190/#192 raw direct-vs-gateway-vs-tunnel SSE proof classification.
- source/proof-map fix: added `tests/cross_matrix/run_responses_raw_sse_parity_contract.py` plus unit tests. The classifier reconstructs authoritative function-call args from `response.function_call_arguments.delta/done` and detects lost gateway/tunnel args separately from an empty final `output_item.done.item.arguments`.
- current artifact: `build/current-responses-raw-sse-parity-20260609.json`, `status=open`, `missing_captures=[direct,gateway,tunnel]`; this is intentional because no live raw captures were supplied in this slice.
- validation: parity classifier tests passed `4/4`; current-suite source-hash guard passed; release-manifest source-hash mirror passed; py_compile and `git diff --check` passed.
- boundary: not a live tunnel proof and not issue closure. Pass requires direct local server, panel gateway, and tunnel raw SSE captures with matching authoritative arguments.

## CODEX - 2026-06-09 Gemma4 cross-shard MoE sidecars and audio waveform source fix
- continued from concurrent edits without reverting them.
- blocker reduced: #191/#188 Gemma4 QAT/native MXFP4 loader/media source correctness for MoE bundles and Gemma4 audio inputs.
- source fix: Gemma4 MoE native-MXFP expert sidecars now hydrate `.scales`/`.biases` from the safetensors index when packed expert weights and sidecars land in different shards, then split/dequant into runtime `experts.switch_glu.{gate,up,down}_proj.weight`.
- source fix: Gemma4/Gemma4 Unified audio requests now decode temp WAV paths into float32 waveform arrays before calling processors that expect raw waveform arrays.
- source fix: `_run_vision_encoding_inner` now uses a fallback request id for request-like internal test/probe objects that do not expose `request_id`, preserving MiMo/media prefill guard telemetry without crashing unit fixtures.
- proof artifact: `build/current-model-artifact-format-contract-after-gemma4-cross-shard-sidecars-audio-waveform-20260609.json`, `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest rc=0 passed=179 deselected=192`.
- boundary: source/no-heavy and unit decode only. No Gemma4 26B live memory-safe proof, no audio semantic live proof, no installed-app proof, and no release/sign/package action.

## CODEX - 2026-06-09 Gemma QAT inventory row normalization
- blocker reduced: Gemma QAT/native MXFP4 release-gate accuracy after local Gemma4 E2B/E4B/12B/26B/31B QAT bundles appeared under `/Users/eric/models/JANGQ-AI`.
- source/proof-map fix: normalized stale Gemma3n E2B/E4B inventory/checklist row IDs to `gemma4_e2b_qat_native_mxfp4` and `gemma4_e4b_qat_native_mxfp4`; full checklist failure names now use Gemma4 E2B/E4B.
- tightened matching: Gemma4 12B/26B/31B QAT rows require `qat` and `mxfp4` path markers, so older JANG/MTP or non-QAT MXFP rows cannot satisfy QAT/native proof rows.
- proof artifact refreshed: `build/current-gemma-qat-native-mxfp4-local-inventory-20260609.json`, `status=open`, `count=16`, `missing_required_rows=[]`, open rows exactly the five Gemma4 QAT/native MXFP4 rows.
- checklist runner artifact: `build/current-full-release-objective-checklist-after-gemma-qat-inventory-row-correction-20260609.json`, still `status=open`, `failed_count=130`.
- validation: Gemma inventory/checklist focused tests passed `4/4`; current-suite Gemma proof-map tests passed `3/3`; release-manifest source-hash mirror tests passed `2/2`; py_compile and `git diff --check` passed.
- boundary: no model load, no live Gemma clearance, no package/sign/notarize/release. Live API/UI/cache/media/installed-app proof remains required.
- now: downloaded all five JANGQ-AI Gemma QAT/native MXFP4 repos locally and corrected the Gemma QAT inventory gate to track the actual Gemma4 E2B/E4B row IDs instead of stale Gemma3n names.
- local paths: `/Users/eric/models/JANGQ-AI/gemma-4-E2B-it-qat-MXFP4`, `/Users/eric/models/JANGQ-AI/gemma-4-E4B-it-qat-MXFP4`, `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4`, `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-MXFP4`, `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-MXFP4`.
- inventory artifact: `build/current-gemma-qat-native-mxfp4-local-inventory-20260609.json`, `status=open`, `missing_required_rows=[]`, `open_required_rows=[gemma4_e2b_qat_native_mxfp4, gemma4_e4b_qat_native_mxfp4, gemma4_12b_native_mxfp4, gemma4_26b_vl, gemma4_31v_or_31b_vl]`.
- release boundary: downloads and no-heavy inventory only. Live multiturn/tool/parser/cache/media/UI/installed-app proof remains open for every QAT row; do not sign/notarize/release from this state.

## CODEX - 2026-06-09 Gemma4 QAT MXFP4 PLE loader status
- now: source fix in progress/verified for Gemma4 QAT/native MXFP4 PLE tensors. Quantized Gemma4 PLE modules keep packed MXFP `uint32` weights and `uint8` scales; plain/non-quantized PLE modules use native MXFP dequant instead of affine fallback.
- proof: `build/current-model-artifact-format-contract-after-gemma4-qat-mxfp4-ple-preserve-20260609.json`, `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest passed=175`.
- live E2B smoke: `build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-tools-image-after-quantized-ple-preserve-20260609/JANGQ_gemma-4-E2B-it-qat-MXFP4/result.json` loaded and served. Prior load/runtime PLE failures are gone; remaining one failure is exact tool-result punctuation (`STORED blue-cat` vs expected `STORED blue-cat.`).
- next agent should continue from that single E2B exactness failure, then run E4B/12B/26B/31B QAT live API/cache/media rows and installed-app/UI parity. Do not mark Gemma QAT rows pass from this partial proof.
- release boundary: no package/sign/notarize/tag/download/release work.
- follow-up: `_fix_quantized_bits()` now also preserves native MXFP modules with `uint8` scales during post-load repair; proof `build/current-model-artifact-format-contract-after-native-mxfp-scale-preserve-20260609.json`, `status=pass`, `model_artifact_format_pytest passed=176`.
- follow-up: Gemma4 MoE fused native-MXFP experts now split/dequant to runtime `switch_glu` float weights; proof `build/current-model-artifact-format-contract-after-gemma4-moe-mxfp-expert-split-20260609.json`, `status=pass`, `model_artifact_format_pytest passed=177`.
- 26B A4B QAT smoke boundary: `build/current-all-local-model-smoke-gemma4-26b-a4b-qat-mxfp4-tools-image-after-moe-mxfp-expert-split-20260609/JANGQ_gemma-4-26B-A4B-it-qat-MXFP4/result.json` loaded then first prefill hit Metal OOM at about `53.8GB` active baseline. Treat as memory/preflight open row, not parser/tool clearance.
- 12B QAT smoke boundary: `build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-tools-image-after-native-mxfp-fixes-20260609/JANGQ_gemma-4-12B-it-qat-MXFP4/result.json` loaded/served as `gemma4_unified`; one failure remains, same `STORED blue-cat` punctuation exactness row as E2B/E4B.
- 31B QAT smoke boundary: `build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-image-after-native-mxfp-fixes-20260609/JANGQ_gemma-4-31B-it-qat-MXFP4/result.json` loaded/served; one failure remains, same `STORED blue-cat` punctuation exactness row as E2B/E4B/12B.

## CODEX - 2026-06-09 04:58 PDT Gemma4 upstream video processor kwargs patch
- now: added and verified `vmlx_engine/runtime_patches/mlx_vlm_compat.py` for upstream `mlx-vlm` PR #1321.
- blocker reduced: #191/#188 Gemma4 QAT/native MXFP4 `media` processor construction. Pinned `mlx_vlm` rejected standard HF video processor keys because `Gemma4VideoProcessor.__init__` had no `**kwargs`.
- source fix: runtime patch filters unknown HF video keys (`do_convert_rgb`, `do_sample_frames`, `resample`, `return_metadata`) while preserving accepted Gemma4 video settings; auto-installed from `runtime_patches/__init__.py`.
- proof: `tests/test_mlx_lm_runtime_patches.py` passed `5/5`; focused current-suite/install/release hash guard set passed `10/10`; `mlx_vlm_compat.py` is now in source-hash boundaries.
- other-agent list: `mlx-lm` #1327 short-prompt think-token clamp, `mlx-vlm` #1332 Qwen3-VL deepstack chunked-prefill alignment, and `mlx-vlm` #1328 LFM2.5 VL loading are mapped candidates only; require local repro/mapping before port.
- boundary: source/no-heavy only. No Gemma live media clearance, installed app proof, release, signing, notarization, tag, or downloads.

## CODEX - 2026-06-09 Responses raw SSE parity strict expected-args gate
- blocker reduced: #190/#192 Responses direct/gateway/tunnel raw SSE proof quality.
- source fix: `tests/cross_matrix/run_responses_raw_sse_parity_contract.py` now supports expected function name, expected authoritative arguments, parse-clean checks, and `--require-reasoning-events`.
- current-suite default: raw SSE parity now requires `lookup`, `{"query":"alpha"}`, and reasoning summary events so a no-reasoning-disable workaround cannot satisfy the row.
- proof: parity classifier tests and current-suite marker guard passed `7/7`; regenerated artifact remains `status=open`, `missing_captures=[direct,gateway,tunnel]`, with the stricter `expected` block recorded.
- boundary: no live direct/gateway/tunnel captures were available on this host in this slice; do not close #192 or #190 from this. Need real raw SSE captures across all three surfaces.

## CODEX - 2026-06-09 Gemma4 audio capability requires tower weights
- now: tightened `model_type=gemma4` audio capability detection to require actual `audio_tower.*` weights in `model.safetensors.index.json`; token-only `audio_token_id` no longer advertises audio.
- blocker reduced: Gemma4 QAT/native MXFP4 advertised `media` honesty for 12B/26B/31B-style bundles.
- proof: focused runtime modality tests passed `3/3`; `py_compile` and `git diff --check` passed.
- boundary: capability gate only. No live audio semantic clearance, installed-app/UI proof, package, signing, notarization, tag, download, or release readiness.

## CODEX - 2026-06-09 reasoning parser package/hash parity
- blocker reduced: `parser/template` package/runtime drift for registered reasoning parsers used by Qwen3/N2, Gemma4, MiniMax M2, GPT-OSS, Mistral, DeepSeek R1, and think/XML thinking paths.
- root cause: package/install/current-suite hash surfaces protected only `vmlx_engine/reasoning/__init__.py` and `think_xml_parser.py` while `reasoning/__init__.py` registers the wider parser matrix.
- source fix: added every top-level `vmlx_engine/reasoning/*.py` file to bundled-python verifier, release-gate, staged packaged integrity, installed-app runtime parity, and current-suite source hash lists.
- regression coverage: installed-app parity, packaged integrity, release-gate, engine audit, current-suite, and release-manifest tests now assert every top-level reasoning parser file is covered.
- red/green proof: five focused guard tests failed before manifest wiring on missing reasoning parser files, then passed `5/5`; the engine audit script assertion passed `1/1`; `bash -n panel/scripts/verify-bundled-python.sh`, `py_compile`, and `git diff --check` passed.
- boundary: package/source-hash parity only. This does not clear live Gemma QAT/native rows, N2 JANG_1L runtime/cache/API/UI proof, MiMo exactness/media/tool rows, raw direct/gateway/tunnel SSE parity, installed-app behavior, or any release/signing step.

## CODEX - 2026-06-09 N2 JANGTQ2 live proof objective consumption
- blocker reduced: N2 release-board accuracy for current-source JANGTQ2 chat/cache/Responses proof.
- source/proof-map fix: `summarize_objective_proof.py`, current regression suite, full checklist, and release manifest now point at `build/current-objective-proof-after-n2-jangtq2-live-proof-20260609.json`; the N2 objective row includes `build/current-n2-jangtq2-chat-cache-responses-proof-after-responses-parser-20260609.json`.
- proof fields recorded: `status=pass`, `stable_text=true`, tool/Responses/stream probes pass, `cache_hit_cache_detail=paged+ssm`, `cache_hit_cached_tokens=8`, block-disk writes/hits present, and SSM disk stores present.
- boundary: N2 remains `OPEN`. This does not clear JANG_1L runtime/cache/API/UI, media, installed-app/UI, same-model tunnel parity, fresh-process L2 restart, package, signing, notarization, tag, download, or release readiness.

## CODEX - 2026-06-09 N2 JANGTQ2 fresh-process L2 objective consumption
- live proof: `build/current-n2-jangtq2-chat-cache-responses-l2-proof-20260609.json`, `status=pass`; ran one bounded current-source N2 JANGTQ2 gate with `--include-l2-restart-probe` and `--min-available-gb 96`.
- restart evidence: `l2_restart_probe_pass=true`, visible `ACK`, `restart_cached_tokens=8`, `restart_cache_detail=paged+ssm+disk`, `block_disk_hits=1`, and `ssm_disk_hits=1`.
- proof-map fix: objective digest, current regression suite, full checklist, and release manifest now point at `build/current-objective-proof-after-n2-jangtq2-l2-live-proof-20260609.json`; N2 still reports `status=open`.
- boundary: current-source N2 JANGTQ2 only. No JANG_1L, installed-app/UI, media, same-model tunnel, package, signing, notarization, tag, download, or release clearance.

## CODEX - 2026-06-09 MiMo current-evidence objective cleanup
- blocker reduced: MiMo release-board accuracy and stale proof noise.
- proof-map fix: MiMo objective evidence now uses current present artifacts only: structural verify, `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`, and `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json`.
- result: regenerated `build/current-objective-proof-after-n2-jangtq2-l2-live-proof-20260609.json`; MiMo `missing_evidence=[]`, `current_evidence_missing=[]`, cache-vs-no-cache `status=pass`, row remains `OPEN`.
- boundary: no MiMo release clearance. Exactness, decode speed, VL/audio/video wiring/E2E proof, JANGTQ2/JANG_2L media/L2, UI, installed app, package, signing, notarization, tag, download, and release remain open.

## CODEX - 2026-06-09 Gemma QAT source-smoke objective detail
- blocker reduced: Gemma QAT/native MXFP4 release-board traceability for other-agent source-smoke proofs.
- proof-map fix: Gemma objective details now list the five exact source-smoke summary artifacts and media-backing facts from `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`.
- result: regenerated `build/current-objective-proof-after-n2-jangtq2-l2-live-proof-20260609.json`; Gemma QAT row remains `OPEN`, with `all_required_source_live_smokes_present=true` and `all_required_live_proofs_present=false`.
- boundary: source-smoke traceability only. Full live Responses/tool/media/cache/UI/installed-app/tunnel proof, package, signing, notarization, tag, download, and release remain open.

## CODEX - 2026-06-09 full checklist refresh after objective details
- refreshed `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` from current source.
- result: `status=open`, `failed_count=122`; stale N2/MiMo details are gone from the regenerated artifact, and Gemma QAT source-smoke/media-backing objective detail is now visible downstream.
- boundary: no release action. Responses tunnel/reasoning, N2 JANG_1L, MiMo exactness/media, Gemma full live/UI/tunnel, Qwen proof gaps, DSV4, package, signing, notarization, tag, and download rows remain open.

## CODEX - 2026-06-09 Gemma QAT objective detail guard
- blocker reduced: Gemma QAT/native MXFP4 release-board evidence drift.
- integrated in-flight objective digest coverage for the five Gemma4 QAT/native MXFP4 rows: E2B, E4B, 12B, 26B, and 31B.
- guard now asserts the objective details expose exact source-smoke artifacts for all five rows and media-backing truth: E2B/E4B audio tower backed, 12B audio embed-only with vision backed and video proof required.
- validation: `tests/test_objective_proof_digest.py::test_objective_proof_digest_tracks_gemma_qat_native_mxfp4_release_blocker` passed; `py_compile` and `git diff --check` passed for the touched test/source surfaces.
- boundary: proof-map/coverage only. No Gemma QAT release clearance, no model launch, no installed-app/UI/tunnel proof, no package/sign/notarize/tag/download.

## CODEX - 2026-06-09 Responses reasoning tool-args source boundary
- blocker checked: reported Responses SSE heartbeat/tool_call_generating path where `tc_args` is empty when reasoning is enabled.
- current source behavior: `stream_responses_api` parses accumulated content, accumulated reasoning, stripped full text, and full text, then prefers parsed tool calls with non-empty arguments. It does not synthesize missing args and does not rely on disabling reasoning.
- validation: focused source tests passed for heartbeat buffering preserving final tool args and reasoning-channel tool calls preserving args: `tests/test_server.py -k "streaming_responses_tool_call_arguments_survive_buffering or streaming_responses_reasoning_tool_call_keeps_arguments"`; `py_compile` passed for `vmlx_engine/server.py` and related tests.
- boundary: this is source-unit evidence only. Same-model direct/gateway/tunnel raw SSE capture with the reported model/request is still required before closing #190/#192 or calling deployed app/API fixed.

## CODEX - 2026-06-09 Responses tunnel model availability classifier
- blocker reduced: #190/#192 same-model raw SSE parity classification for the public tunnel surface.
- source/proof-map fix: `tests/cross_matrix/run_responses_raw_sse_parity_contract.py` now parses model lists from raw JSON `model_not_found` errors and records whether the expected tunnel model is actually advertised. The full release checklist now exposes this as `responses_raw_sse_parity_tunnel_expected_model_advertised`.
- refreshed proof: `build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-e2b-after-parser-20260609.json`, `status=fail`, `missing_captures=[]`, `checks.tunnel_expected_model_advertised=false`; tunnel returned `model_not_found` for `gemma4-e2b-sse` and the parsed available-model list does not include it.
- checklist: `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`, `status=open`, `failed_count=123`; new failed row is `responses_raw_sse_parity_tunnel_expected_model_advertised`.
- boundary: direct/gateway Gemma4 E2B argument streaming remains separately proven, but direct/gateway still have `reasoning_events=0` and tunnel is not same-model. Next agent should serve/target the same model through tunnel, recapture direct/gateway/tunnel with reasoning events enabled, and keep missing required XML args fail-closed rather than inventing params from preamble text.

## CODEX - 2026-06-09 Gemma QAT source-video proof consumption
- blocker reduced: Gemma QAT/native MXFP4 source-proof tracking for 12B/26B/31B video runtime.
- source/proof-map fix: `tests/cross_matrix/run_gemma_qat_native_mxfp4_inventory_gate.py` now extracts `vl_blue_video` and `text_no_media_after_video` from each source-smoke summary and records `video_runtime_proven` plus `post_video_text_recovery_proven`; objective/checklist details consume these fields.
- refreshed proof: `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`, `status=open`, `missing_required_rows=[]`, `source_live_smoke_open_rows=[]`, and `gemma4_{12b,26b,31v_or_31b}_video_runtime_source_proven=true`.
- checklist: `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`, `status=open`, `failed_count=120`; the three Gemma video-runtime subrows moved green, but `gemma_qat_native_mxfp4_all_live_proofs_present` remains red.
- boundary: current-source video smoke proof only. This does not clear installed-app/UI/tunnel parity, full Responses stream/tool-argument proof, broader API/cache release proof, package, signing, notarization, tag, or downloads.

## CODEX - 2026-06-09 MiMo cache-vs-no-cache classifier consumption
- blocker reduced: MiMo no-source exactness classifier proof-map accuracy for cache/KV/L2 exclusion.
- source/proof-map fix: `tests/cross_matrix/run_mimo_v2_no_source_exactness_classifier.py` now consumes `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json`. It only excludes cache/KV/L2 as primary when no-cache, warm-store, and cache-hit modes are HTTP 200, top-10 logprobs match, cache-hit tokens are present, and cache detail is paged.
- refreshed classifier: `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`, `status=open`, `classification=jangtq2_plain_literal_copy_fails_before_parser_or_json_repair`, `excluded_surfaces.prefix_paged_l2_or_kv_quant_primary_cause=true`.
- checklist: `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`, `status=open`, `failed_count=119`; `mimo_no_source_classifier_excludes_parser_cache_sampling` moved green.
- boundary: MiMo remains release-open. Do not chase parser/cache/L2/sampling as primary for the current JANGTQ2 literal exactness failure without new contrary logits; remaining action is artifact/logit diagnosis, corrected quantization contract, or runtime decode proof.

## CODEX - 2026-06-09 Gemma4 QAT JANG_4M coverage note
- added release-scope requirement: Gemma 4 QAT `JANG_4M` bundles are separate from native `MXFP4` QAT bundles and must be proven separately.
- required coverage for QAT `JANG_4M`: runtime autodetect, loader/sidecar handling, model-owned generation defaults, Gemma4 tool/reasoning parser selection, mixed-SWA cache, prefix cache, TurboQuant KV boundary where valid, block-disk L2 restore, Responses streaming/content delta/tool args, chat multi-turn, image/video/audio capability honesty, UI/CLI settings parity, and installed-app parity.
- boundary: MXFP4 QAT source smokes do not clear QAT `JANG_4M`; QAT `JANG_4M` source/load proof does not clear MXFP4. Both need explicit live proof before release.

## CODEX - 2026-06-09 Qwen35 tunnel raw SSE recapture
- now: Qwen35 public tunnel raw Responses SSE was recaptured against advertised model `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP` with reasoning enabled and required `record_fact`. New raw capture `build/responses-sse-captures-20260609/tunnel-qwen35-mxfp8-mtp-tool-recapture-max512-20260609.sse` preserves args via `response.function_call_arguments.delta/done` and final item args `{"value": "blue-cat"}`, has `reasoning_events=10`, reports same model, and parse errors are `0`, but still reuses `output_index=0` for both `message` and `function_call`.
- proof: `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-recapture-20260609.json`, `status=fail`; `all_present_surfaces_have_authoritative_args=true`, `all_present_surfaces_have_required_reasoning=true`, `all_present_surfaces_have_valid_output_item_indices=false`, conflict `[0]`. The checklist now points at this recapture artifact; `qwen35_raw_sse_valid_output_item_indices` remains red. This is deployed/tunnel output-index freshness, not the empty-args parser failure.
- validation: raw-SSE/checklist focused tests passed `16/16`; `py_compile` and `git diff --check` passed. Boundary: same-model direct/gateway captures are still missing for complete parity; no release/package/sign/notarize action.

## CODEX - 2026-06-09 N2 JANG_1L memory proof refresh
- blocker kept honest: N2 JANG_1L live runtime/cache/API/UI is still `OPEN`, but it now has current preflight and live-gate skip artifacts instead of stale or missing proof.
- current preflight: `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`, `decision=do_not_launch`, `indexed_payload_gib=110.57`, `required_available_gib=118.57`, `available_gib=111.12`, `memory_gap_gib=7.45`.
- current live-gate attempt: `build/current-n2-jang1l-chat-cache-proof-20260609.json`, `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`, `available_gib=111.25`, `memory_gap_gib=7.32`; requested tool, Responses, Responses stream, and L2 restart probes are recorded but skipped before launch.
- proof-map fix: objective digest/checklist/current-suite/release-manifest pointers now use `build/current-objective-proof-after-n2-jang1l-memory-refresh-20260609.json`; full checklist remains `status=open`, `failed_count=112`.
- boundary: no N2 JANG_1L live clearance and no release action. Next safe launch needs available memory at or above `118.57 GiB` or a smaller-runtime strategy; do not lower the JANG_1L guard just to force a run.

## CODEX - 2026-06-09 empty XML tool-call parser fail-closed
- blocker reduced: Qwen/Nemotron-style raw XML tool dialect no longer turns `<tool_call><function=exec_command></function></tool_call>` into executable `arguments={}` in the generic parser.
- source fix: `vmlx_engine/api/tool_calling.py` now requires at least one `<parameter=...>` entry before accepting Nemotron XML function blocks; valid parameterized XML still parses normally.
- validation: new generic parser regressions passed, existing Responses streaming guards passed for argument buffering, required-tool empty XML rejection, no `arguments:"{}"` emission, output-index ordering, and reasoning-channel tool args; Nemotron/Step parser coverage passed.
- boundary: this is a source parser fix/proof only. Same-model deployed direct/gateway/tunnel raw SSE remains open for #190/#192, and no package/sign/notarize/tag/download/release was run.

## CODEX - 2026-06-09 MiMo artifact-diagnosis classifier pointer
- blocker reduced: MiMo proof-map drift between the active release board and the artifact/logit/quant diagnosis lane.
- source/proof-map fix: MiMo no-source classifier, current audit, objective digest, checklist, and release-manifest pointers now use `build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json`.
- refreshed outputs: `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, `build/current-objective-proof-after-n2-jang1l-memory-refresh-20260609.json`, and the full checklist. Checklist remains `status=open`, `failed_count=112`.
- current MiMo finding: `model_upload_action_required=true`; parsed tool structure is valid but literals mutate (`blue-cat`, `B7-CAT-09`, JSON sentinel, tool-result punctuation), so do not chase parser/cache/L2/sampling as primary without contrary logits.
- boundary: no MiMo release clearance. Remaining work is corrected artifact/quantization contract, runtime decode/logit fix, media proof, speed, UI/installed-app parity, and release packaging only after explicit release action.

## CODEX - 2026-06-09 Qwen35 required-tool/cache proof
- blocker reduced: Qwen3.6 35B MXFP8-MTP Responses required-tool + historical tool-result + hybrid cache row.
- source fix: nonstream Chat Completions and Responses now do one bounded real-generation retry when `tool_choice=required` produces prose and no tool call; the retry keeps the same tool schema/choice and disables thinking for the correction turn. If the retry still has no tool call, the existing 400 fail-closed behavior remains.
- gate fix: `tests/cross_matrix/run_responses_long_tool_cache_gate.py` now asks tools-enabled turns for a tool call as the first assistant output and sends a user continuation after `function_call_output` requiring `TOOL_EVIDENCE: <exact path:line>`.
- live proof: `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-after-historical-tool-required-20260607/SUMMARY.json`, `overall_pass=true`; turn 1 and 2 required tool calls were produced and grounded, turn 2/3 cached tokens were `128`/`256`, cache detail was `paged+ssm`, and no HTTP/tool-markup/loop failures were recorded.
- checklist: `build/current-full-release-objective-checklist-after-qwen35-long-tool-cache-20260609.json`, `status=open`, `failed_count=73`; every `qwen35_long_tool_cache_*` row is green, while Qwen35 restart/installed-video rows and wider release rows remain open.
- validation: focused required-tool/tool-cache tests passed, tool-format required tests passed, `py_compile` passed, and the live Qwen35 gate passed against a fresh server with native MTP/hybrid SSM/TurboQuant/block-L2 enabled.
- boundary: no package, signing, notarization, tag, download, or release step was run. Same-model direct/gateway/tunnel raw SSE for #190/#192, N2 JANG_1L live clearance, MiMo exactness/media, Gemma full live/UI, installed-app parity, and release packaging remain open.

## CODEX - 2026-06-09 Qwen35 fresh-process L2 restore proof
- blocker reduced: Qwen3.6 35B MXFP8-MTP restart/L2 restore row.
- source/proof harness: added `tests/cross_matrix/run_qwen35_mxfp8_mtp_restart_l2_restore.py`, a two-process live gate that writes the checklist's `phases.phase1/phase2` schema. Phase 1 writes block+SSM L2; phase 2 starts a fresh server on the same cache dir and verifies disk-backed hybrid SSM restore with native MTP still active.
- live proof: `build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json`, `status=pass`; phase 1 HTTP 200, `block_disk_cache.disk_writes=1`, `ssm_companion_disk.stores=1`; phase 2 HTTP 200, `cached_tokens=8`, `cache_detail=paged+ssm+disk`, `block_disk_cache.disk_hits=1`, `ssm_companion_disk.hits=1`, native cache `hybrid_ssm_v1/hybrid_ssm_typed`, MTP `native_runtime_active`, depth 3.
- checklist: `build/current-full-release-objective-checklist-after-qwen35-restart-l2-20260609.json`, `status=open`, `failed_count=67`; all `qwen35_restart_*` rows are green.
- validation: new Qwen35 restart runner tests passed `3/3`, `py_compile` passed, live two-process gate passed, and the full checklist consumed the proof.
- boundary: Qwen35 source rows now have startup, long-tool/cache, and restart/L2 proof. Qwen35 installed-app/video/UI rows remain open, as do Responses tunnel parity, N2 JANG_1L live clearance, MiMo exactness/media, Gemma full live/UI, package parity, signing, notarization, tag, downloads, and release.

## CODEX - 2026-06-09 Gemma4 QAT JANG_4M proof-map lane
- blocker reduced: `Gemma4 QAT JANG_4M` release proof-map/inventory tracking, no heavy model launch.
- source change: `tests/cross_matrix/run_gemma_qat_native_mxfp4_inventory_gate.py` now tracks `gemma4_12b_qat_jang4m` as a separate open row from native MXFP4 QAT rows.
- regenerated no-heavy artifacts: `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`, `build/current-objective-proof-after-n2-jang1l-memory-refresh-20260609.json`, and `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`.
- inventory result: `status=open`, `missing_required_rows=[]`, `open_required_rows` includes `gemma4_12b_qat_jang4m`, E2B/E4B/12B native MXFP4, 26B VL, and 31V/31B VL.
- boundary: Gemma4 QAT JANG_4M remains open. The existing no-media source smoke is `probe_failed`; release proof still requires autodetect, model-owned `generation_config`, Gemma4 tool/reasoning parser, mixed-SWA/prefix cache, TurboQuant KV boundary where valid, block-disk L2, Responses streaming args/content deltas, media honesty, UI/CLI parity, and installed-app parity.
- focused validation: `uv run pytest tests/test_gemma_qat_native_mxfp4_inventory_gate.py tests/test_objective_proof_digest.py::test_objective_proof_digest_tracks_gemma_qat_native_mxfp4_release_blocker tests/test_full_release_objective_checklist.py::test_full_release_objective_checklist_blocks_open_gemma_qat_jang4m_row -q` passed `10/10`.

## CODEX - 2026-06-09 N2/MiMo focus correction
- blocker reduced: MiMo V2.5 media/runtime truthfulness for the N2/MiMo release lane; no release/sign/package action.
- source/proof-map fix: `tests/cross_matrix/run_mimo_v2_jang2l_current_audit.py` no longer treats source media component presence as live runtime media support. It records `source_media_components_present` separately and only clears `runtime_media_wired` / `media_runtime_implementation` when runtime capabilities or live E2E prove media support.
- stale detector fix: the raw-audio bridge check now follows the current source split, where `audio=all_audio if all_audio else None` lives in `vmlx_engine/engine/batched.py` and the batch generator handles raw audio processing/processor forwarding.
- refreshed MiMo audit: `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, `status=open`; media weights and source components are present, raw audio ingestion is present, runtime capabilities are text-only, `runtime_media_wired=false`, and `media_runtime_implementation=false`.
- refreshed classifier/checklist: `build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json` remains `status=open`; side checklist `build/current-full-release-objective-checklist-after-mimo-media-runtime-boundary-20260609.json` remains `status=open`, `failed_count=72`, with `mimo_media_runtime_implementation` and `mimo_mimo_media_wired` explicitly red.
- N2 current boundary rechecked this turn: available memory was about `111.16 GiB`, still below the JANG_1L live-gate requirement of `118.57 GiB`; do not lower the guard just to launch Nex/N2 Pro 397B.
- validation: MiMo audit/classifier/checklist focused tests passed `36/36`; `py_compile` passed for the edited MiMo audit files.
- boundary: MiMo is not release-cleared. Remaining MiMo work is real VL/audio/video runtime support and live media/L2 proof, artifact/logit/quant exactness or decode fix, decode speed target, UI/installed-app parity. N2 Pro 397B remains open until memory-safe JANG_1L live proof or a real smaller-runtime strategy exists.

## CODEX - 2026-06-09 Gemma4 E2B QAT JANG_4M source smoke
- blocker reduced: Gemma4 QAT JANG_4M source-live text/tool/cache/L2 coverage for the smallest downloaded bundle.
- proof: `build/current-all-local-model-smoke-gemma4-e2b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-E2B-it-qat-JANG_4M/result.json`, `status=pass`; summary at `build/current-all-local-model-smoke-gemma4-e2b-qat-jang4m-tools-nomedia-l2-20260609/summary.json`.
- proven surfaces: autodetect `model_type=gemma4`, tool/reasoning parser `gemma4`, model-owned launch defaults with `--default-enable-thinking false`, visible ACK repeat, multi-turn `blue cat`, reasoning separation, required `record_fact({"value":"blue-cat"})`, tool-result continuation `STORED blue-cat.`, exact JSON/code probes, mixed-SWA cache hit `cached_tokens=56` with `cache_detail=paged+mixed_swa`, block-disk writes, and fresh-process L2 restart `pass`.
- updated artifacts: Gemma inventory/objective/checklist now consume the E2B QAT JANG_4M source smoke; `source_live_smoke_open_rows` still lists E4B, 12B, 26B, and 31B QAT JANG_4M.
- boundary: no Gemma4 QAT JANG_4M release clearance. E2B media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, and larger QAT JANG_4M bundles remain open.

## CODEX - 2026-06-09 Gemma4 E4B QAT JANG_4M source smoke
- blocker reduced: Gemma4 QAT JANG_4M source-live text/tool/cache/L2 coverage for E4B.
- proof: `build/current-all-local-model-smoke-gemma4-e4b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-E4B-it-qat-JANG_4M/result.json`, `status=pass`; summary at `build/current-all-local-model-smoke-gemma4-e4b-qat-jang4m-tools-nomedia-l2-20260609/summary.json`.
- proven surfaces: visible ACK repeat, multi-turn recall, reasoning separation, required `record_fact({"value":"blue-cat"})`, tool-result continuation, exact JSON/code probes, mixed-SWA cache hit `cached_tokens=56` with `cache_detail=paged+mixed_swa`, block-disk writes, and L2 restart with `disk_hits=2`.
- updated artifacts: Gemma inventory/objective/checklist now consume E2B and E4B QAT JANG_4M source smokes; `source_live_smoke_open_rows` is now 12B, 26B, and 31B QAT JANG_4M.
- boundary: no Gemma4 QAT JANG_4M release clearance. Media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, and remaining larger QAT JANG_4M bundles remain open.

## CODEX - 2026-06-09 Gemma4 12B QAT JANG_4M source smoke blocker
- blocker classified: Gemma4 12B QAT JANG_4M tool-call visible sentinel leak.
- proof: `build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-12B-it-qat-JANG_4M/result.json`, `status=probe_failed`.
- concrete failure: `tool_required` returned a valid native `record_fact({"value":"blue-cat"})` call but also visible content `<audio|>`, so the gate failed `tool_visible_text_leak`.
- non-failing surfaces in same run: mixed-SWA cache hit and L2 restart passed; L2 summary had `disk_hits=2`, `cache_hit_tokens=56`.
- initial classification: parser/template/special-token leak on `gemma4_unified`; not a cache/L2 failure and not missing generation config. Do not hide by accepting visible `<audio|>` as normal text.
- updated artifacts: Gemma inventory/objective/checklist now point 12B QAT JANG_4M at this current failure artifact instead of the older non-QAT 12B JANG_4M proof.

## CODEX - 2026-06-09 Gemma4 26B QAT JANG_4M source smoke
- blocker reduced: Gemma4 QAT JANG_4M source-live text/tool/cache/L2 coverage for 26B.
- proof: `build/current-all-local-model-smoke-gemma4-26b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-26B-A4B-it-qat-JANG_4M/result.json`, `status=pass`; summary at `build/current-all-local-model-smoke-gemma4-26b-qat-jang4m-tools-nomedia-l2-20260609/summary.json`.
- proven surfaces: visible ACK repeat, multi-turn recall, reasoning separation, required tool call, tool-result continuation, exact JSON/code probes, mixed-SWA cache hit `cached_tokens=56` with `cache_detail=paged+mixed_swa`, block-disk writes, and L2 restart with `disk_hits=2`.
- updated artifacts: Gemma inventory/objective/checklist now consume E2B, E4B, and 26B QAT JANG_4M source smokes; source-smoke open rows are 12B and 31B.
- boundary: no release clearance; 12B has current `<audio|>` tool-visible leak, 31B still needs source smoke, and all media/UI/installed-app/Responses raw SSE release rows remain open.

## CODEX - 2026-06-09 Qwen35 tunnel raw SSE output-index proof
- blocker classified: Responses raw SSE same-model/tool/reasoning parity for Qwen35 tunnel.
- proof: `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-20260609.json`, `status=fail`.
- positive evidence: existing tunnel capture `build/responses-sse-captures-20260609/tunnel-qwen35-mxfp8-mtp-tool-20260609.sse` has reasoning events (`8`), function-call argument deltas (`2`), function-call arguments done (`1`), expected tool `record_fact`, and authoritative args `{"value": "blue-cat"}` for model `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`.
- failure boundary: the same tunnel capture emits both `message` and `function_call` at `output_index=0`, so `all_present_surfaces_have_valid_output_item_indices=false`. This is not an `arguments:{}` failure and not a reasoning-disable workaround.
- next proof: capture Qwen35 direct local and gateway raw SSE for the same served model and prove `message:0`, `function_call:1` or no message item before function-call. Keep Gemma E2B tunnel wrong-model and Qwen35 tunnel output-index as separate blockers.
- release boundary: no release/sign/package action. Responses raw SSE remains open for same-model direct/gateway/tunnel parity, required reasoning events, valid output indices, content/tool arg delta/done, and tool-result continuation across Gemma, Qwen/N2, MiMo, and installed app surfaces.

## CODEX - 2026-06-09 Gemma4 31B QAT JANG_4M source smoke
- blocker reduced: Gemma4 QAT JANG_4M source-live text/tool/cache/L2 coverage for 31B.
- proof: `build/current-all-local-model-smoke-gemma4-31b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-31B-it-qat-JANG_4M/result.json`, `status=pass`; summary at `build/current-all-local-model-smoke-gemma4-31b-qat-jang4m-tools-nomedia-l2-20260609/summary.json`.
- proven surfaces: visible ACK repeat, multi-turn recall, reasoning separation, required tool call, tool-result continuation, exact JSON/code probes, mixed-SWA cache hit `cached_tokens=56` with `cache_detail=paged+mixed_swa`, block-disk writes, and L2 restart with `disk_hits=2`.
- updated artifacts: Gemma inventory/objective/checklist now consume E2B, E4B, 26B, and 31B QAT JANG_4M source smokes; the only QAT JANG_4M source-smoke open row is 12B due visible `<audio|>` leak in required-tool output.
- boundary: no release clearance; media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, and the 12B unified leak remain open.

## CODEX - 2026-06-09 Gemma4 12B QAT JANG_4M tool sentinel source fix
- blocker reduced: Gemma4 12B QAT JANG_4M required-tool visible `<audio|>` sentinel leak.
- source fix: `vmlx_engine/server.py` now drops exact singleton Gemma modality sentinels (`<audio|>`, `<|audio|>`, image/video variants) from the final visible assistant content only when a structured tool call is present. This extends the existing `thought`/`analysis` channel-marker guard and does not suppress real prose around a tool call or any no-tool response.
- root-cause boundary: `Gemma4ToolParser` already strips the same sentinel family during structural extraction; the server guard covers response-assembly/parser-selection bypass residue without accepting arbitrary visible text as normal.
- validation: `tests/test_gemma4_tool_parser.py` passed `11/11`; `tests/test_engine_audit.py -k 'tool_visible_channel_marker or xml_function_tool_fallback_accepts_native_mimo_schema'` passed `2/2`; `tests/test_all_local_model_smoke.py -k tool_visible_text_leak` passed `1/1`; `py_compile` passed for the touched Python files.
- proof boundary: the parser-fix entry below carries the live 12B rerun that turns the no-media source row green; this server guard is additional source defense against parser-selection/assembly residue. No package, sign, notarize, tag, download, or release action was run.

## CODEX - 2026-06-09 Gemma4 modality sentinel tool-leak fix
- runtime/parser fix: `vmlx_engine/tool_parsers/gemma4_tool_parser.py` now strips exact Gemma4 modality sentinels `<|image|>`, `<image|>`, `<|audio|>`, `<audio|>`, `<|video|>`, and `<video|>` from parser residual content after tool-call extraction.
- regression: `tests/test_gemma4_tool_parser.py::TestGemma4ToolParser::test_native_tool_call_strips_bare_modality_sentinel_leak` failed before the fix with `content='<audio|>'` and passes after.
- live proof: reran 12B QAT JANG_4M after the fix at `build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-tools-nomedia-l2-after-modality-token-clean-20260609/JANGQ_gemma-4-12B-it-qat-JANG_4M/result.json`, `status=pass`.
- result: all five QAT JANG_4M source no-media tool/cache/L2 smokes now pass: E2B, E4B, 12B, 26B, 31B. Inventory has `source_live_smoke_open_rows=[]` and `all_required_source_live_smokes_present=true`.
- boundary: this is not full Gemma release clearance. Media/video/audio E2E, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, package/signing/notarization remain open.

## CODEX - 2026-06-09 Responses/Qwen35 raw SSE output-index release gate
- blocker reduced: #190/#192 Responses raw SSE proof-map completeness for Qwen/Qwen3.6 tunnel output indices.
- source/proof-map fix: full release checklist now requires `all_present_surfaces_have_valid_output_item_indices` for the generic Responses raw-SSE parity row and adds a Qwen35-specific raw-SSE gate consuming `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-20260609.json`.
- regenerated checklist: `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`, `status=open`, `failed_count=73`; new explicit Qwen failures are `qwen35_raw_sse_status_pass` and `qwen35_raw_sse_valid_output_item_indices` with conflicting index `0` on direct/gateway/tunnel copies of the Qwen35 tunnel capture.
- coordination: wrote `.agents/PARALLEL_RELEASE_LANE_HANDOFF_2026_06_09.md` so the parallel agent can pick up same-model Responses capture/fix work, Gemma media/UI, MiMo, N2, DSV4, and MiniMax without relying on chat context.
- validation: focused full-checklist tests for Responses/Qwen35/Gemma green fixture passed `4/4`; `py_compile` passed. No package, sign, notarize, tag, download, or release action was run.
- follow-up source recheck: `build/current-noheavy-api-cache-contract-after-qwen35-output-index-recheck-20260609.json`, `status=pass`; current source still passes `responses_streaming_tool_call_arguments_and_indexes`, gateway argument streaming, reasoning-empty final-args streaming, and stale-port rejection. This narrows the remaining Qwen35 raw-SSE blocker to live same-model direct/gateway/tunnel recapture or deployed/tunnel staleness, not parser/cache/sampling/generation defaults.

# 2026-06-10 - Gemma 12B JANG4M real dev-app proof

- Reduced blocker class: `api/ui` for Gemma 12B JANG4M source-to-dev-app chat proof.
- Tracked proof summary: `build/current-real-ui-live-model-gemma4-12b-jang4m-dev-app-proof-20260610.json`, `status=pass`.
- Raw dev-app proof captures are under ignored `docs/internal/agent-notes/`: Responses/tools/cache and Chat/image/cache both passed.
- Responses app proof: real Electron dev app launched, real vMLX server loaded `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`, built-in `run_command` tool loop executed twice, visible content streamed, and cache metrics showed `paged+mixed_swa` hits with cached tokens `2687` then `2901`.
- Image app proof: Chat Completions UI path sent an image attachment to the Gemma MLLM path and returned visible `Red`; `media.imageSemanticVerified=true`.
- Cache proof: server controls were visible, native mixed-SWA cache was active, and block-disk L2 writes were present.
- Boundary: this is dev Electron app + remote session proof, not installed-app parity, not local session-manager launch proof, not audio/video, not public tunnel raw SSE, and not package/sign/notarize/tag/download/release clearance.

# 2026-06-10 - N2 JANGTQ2 real dev-app proof remains red on Responses delta streaming

- Reduced blocker class: `api/ui` for Nex/N2 Pro JANGTQ2 dev-app proof.
- Tracked proof summary: `build/current-real-ui-live-model-n2-jangtq2-dev-app-proof-20260610.json`, `status=fail`.
- Raw dev-app proof captures are under ignored `docs/internal/agent-notes/`: default prompt and longer-delta prompt both failed only the `responses_delta_streaming` required surface.
- Positive evidence: real Electron dev app launched, real vMLX server loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`, `/v1/responses` completed two turns, built-in `run_command` executed, and probe files contained `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`.
- Positive cache/runtime evidence: health reported `hybrid_ssm_v1` / `hybrid_ssm_typed`, attention-only TurboQuant KV, native SSM companion state, paged+SSM cache hits, block-disk L2 writes, and SSM companion disk stores. Longer attempt ended around `103807.6 MB` active, `108294.9 MB` peak, `l2_block_tokens_on_disk=4626`, `l2_ssm_tokens_on_disk=23454`.
- Red evidence: both attempts failed `requested Responses API mode but proof did not record responses_delta_streaming surface`; first post-tool visible answer collapsed to `Created`, leaving only one multi-delta visible assistant trace in the longer run.
- Boundary: N2 JANGTQ2 source/API/cache proof remains strong, but dev-app Responses delta streaming is not green. Do not package/sign/notarize/release on this row until raw server SSE vs gateway/dev-app stream trace is compared or fixed.

# 2026-06-10 - N2 JANGTQ2 raw SSE boundary narrows dev-app streaming red row

- Reduced blocker class: `api/ui` for Nex/N2 Pro JANGTQ2 Responses streaming.
- Added focused runner `tests/cross_matrix/run_n2_responses_stream_boundary_probe.py` to launch N2 JANGTQ2 once and capture raw Responses SSE for direct server and panel `ApiGateway` tool + tool-result continuation.
- Proof: `build/current-n2-jangtq2-responses-stream-boundary-20260610.json`, `status=pass`.
- Direct server raw SSE: first request produced required `lookup({"query":"alpha"})`; follow-up completed with `16` output-text deltas and visible `N2_DIRECT_DELTA_ONE is confirmed. N2_DIRECT_DELTA_TWO is confirmed.`
- Panel gateway raw SSE: first request produced required `lookup({"query":"alpha"})`; follow-up completed with `14` output-text deltas and visible `N2_DIRECT_DELTA_ONE confirmed. N2_DIRECT_DELTA_TWO confirmed.`
- Runtime/cache proof in same run: model loaded as `qwen3_5_moe`, `hybrid_ssm_v1`, attention-only TurboQuant KV, native SSM companion, async rederive, paged cache, block L2; final memory about `103804.1 MB` active and `105029.3 MB` peak.
- Boundary: this clears direct/gateway raw SSE content-delta transport for N2 JANGTQ2, but it does not clear the earlier real dev-app chat proof. The dev-app red row is narrowed to renderer/chat tool-loop first post-tool visible trace behavior, where the first app answer collapsed to `Created`.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANG_2L real dev-app proof is red on tool/visible output

- Reduced blocker class: `api/ui` for MiMo V2.5 JANG_2L dev-app proof.
- Tracked proof summary: `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-proof-20260610.json`, `status=fail`.
- Raw dev-app proof capture is under ignored `docs/internal/agent-notes/current-real-ui-live-model-mimo-v25-jang2l-chat-tools-cache-20260610-proof.json`.
- Positive evidence: real Electron dev app launched; real vMLX server loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`; health was `healthy`, `model_type=llm`, quant dispatch `mlx_affine_quantized_matmul`; native MiMo mixed-SWA cache surfaced with prefix, paged cache, and block-disk L2.
- Positive cache evidence: app proof observed `cached_tokens=3407`, `cache_detail=paged`, `l2_block_tokens_on_disk=3544`, `disk_writes=57`, and verified server cache controls.
- Red evidence: proof failed `UI turn ended with empty visible assistant content` and `requested real built-in tools but proof did not record long_tool_loop surface`; no tool probe files were created. First assistant output was low-quality prose, not a valid tool call; second assistant streamed 24 tokens with empty visible content.
- Boundary: MiMo JANG_2L remains the stronger checkpoint candidate than MiMo JANGTQ2, but dev-app tool/visible output is not green. Do not chase cache/L2 for this red row; compare raw chat/tool output against panel app trace next.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANG_2L dev-app Chat follow-up narrowed but still red

- Fixed the panel Chat Completions in-turn tool-result follow-up path to suppress the original explicit single-tool `tool_choice` after a tool call has executed.
- Added tracked proof summary `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-followup-proof-20260610.json`, `status=fail`.
- Positive evidence: real Electron dev app loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`, executed `run_command` through Chat Completions, completed two visible assistant turns, reported no send errors, verified server cache controls, observed `cacheHitTokens=8072`, and wrote block-disk L2 with `l2_block_tokens_on_disk=4580`.
- Narrowed failure: the earlier Chat follow-up `tool_choice='required'` final-answer error is no longer observed.
- Remaining red evidence: MiMo still mutates exact tool arguments and paths. `REAL_UI_LIVE_TOOL_ONE` / `REAL_UI_LIVE_TOOL_TWO` became `REAL_UI_LAND_TOOL_ONE` / `REAL_UI_LAND_TOOL_TWO`, and the model wrote `/tmp/real_ui_land_tool_one.txt` / `/tmp/real_ui_land_tool_two.txt` instead of the configured working-directory probe files.
- Boundary: MiMo JANG_2L remains red for dev-app agentic tool-loop exactness. Cache/L2 is positive; next work is model/runtime/artifact decode or tool-argument exactness, not parser repair or fake argument rewriting.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - N2 JANGTQ2 dev-app image/VL proof green

- Ran real Electron dev-app Nex/N2 Pro JANGTQ2 Chat Completions image proof.
- Added tracked proof summary `build/current-real-ui-live-model-n2-jangtq2-image-proof-20260610.json`, `status=pass`.
- Proven: app persisted the image attachment, server loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2` as `model_type=mllm`, `MEDIA_DIAG` saw one `image_url`, runtime processed `num_images_processed=1`, and the assistant answered `Red` for the red-image semantic probe.
- Cache/runtime evidence in the same run: hybrid SSM cache, attention-only TurboQuant KV, server cache controls, `cache_detail=paged+ssm`, `cached_tokens=18`, `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`, `l2_tokens_on_disk=118`, block-disk `disk_hits=3`, and SSM companion stores `2`.
- Honest media cache boundary: the server skipped prefix/paged cache store for the media prompt itself because media embeddings are path-dependent.
- Still open: N2 audio, N2 video, installed-app parity, public tunnel parity, and N2 JANG_1L memory-safe startup.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - N2 JANGTQ2 dev-app video/VL proof green

- Generated a 1-second 64x64 solid-red MP4 fixture with `ffmpeg` and ran real Electron dev-app Nex/N2 Pro JANGTQ2 Chat Completions video proof.
- Added tracked proof summary `build/current-real-ui-live-model-n2-jangtq2-video-proof-20260610.json`, `status=pass`.
- Proven: app persisted the `video_url` attachment, server decoded the base64 MP4, reported `25 total frames @ 25.0 fps`, extracted `4 frames`, processed `num_images_processed=4`, and the assistant answered `The video shows a solid red screen with no visible movement or change.`
- Cache/runtime evidence in the same run: hybrid SSM cache, attention-only TurboQuant KV, server cache controls, `cache_detail=paged+ssm`, `cached_tokens=18`, `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`, `l2_tokens_on_disk=118`, block-disk `disk_hits=3`, and SSM companion stores `2`.
- Honest media cache boundary: the server skipped prefix/paged cache store for the video prompt because media embeddings are path-dependent.
- Still open: N2 audio, installed-app parity, public tunnel parity, and N2 JANG_1L memory-safe startup.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - N2 JANGTQ2 dev-app audio honestly gated

- Generated a 16 kHz mono WAV saying `audio present` and ran real Electron dev-app Nex/N2 Pro JANGTQ2 Chat Completions audio proof.
- Added tracked proof summary `build/current-real-ui-live-model-n2-jangtq2-audio-proof-20260610.json`, `status=fail`.
- Positive evidence before the audio turn: real model loaded as `mllm`, text chat turns completed, hybrid SSM cache and attention-only TurboQuant KV were active, and cache/L2 remained green with `cache_detail=paged+ssm`, `cached_tokens=18`, `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`, `l2_tokens_on_disk=118`.
- Red evidence: app attempted an audio turn, server `MEDIA_DIAG` saw `input_audio`, and the API returned `400 - /v1/chat/completions received unsupported media modality audio. Supported modalities: text, vision, video.`
- Boundary: this is an honest capability guard, not a crash or cache failure. Do not claim N2 JANGTQ2 audio support in the checkpoint release.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - Gemma 12B QAT MXFP4 dev-app Responses/tools/cache proof green

- Ran real Electron dev-app Gemma 4 12B QAT MXFP4 Responses built-in tool/cache proof.
- Added tracked proof summary `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-dev-app-proof-20260610.json`, `status=pass`.
- Proven: app loaded `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4`, executed `run_command` twice, used Responses tool follow-ups with `previous_response_id`, created probe files containing `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`, and streamed visible deltas on both turns.
- Runtime/cache evidence: MXFP4 affine quantized matmul, Metal NA active, mixed-SWA cache, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=3538`, `l2_block_tokens_on_disk=3588`, block-disk `disk_hits=30`, and `disk_writes=58`.
- Caveat: the second visible answer starts with the plain word `thought`. It is not a raw `<think>` tag or tool/parser markup and the leak gates passed, but keep this visible-final style caveat on the board.
- Still open: Gemma 12B QAT MXFP4 image/video/audio dev-app media, installed-app parity, public tunnel parity, and release signing/notarization.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - Gemma 12B QAT MXFP4 dev-app image/VL proof green

- Ran real Electron dev-app Gemma 4 12B QAT MXFP4 Chat Completions image proof.
- Added tracked proof summary `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-image-proof-20260610.json`, `status=pass`.
- Proven: app persisted the image attachment, server `MEDIA_DIAG` saw one `image_url`, Gemma media fallback ran with `1 image(s)`, and the assistant answered `Red` for the red-image semantic probe.
- Runtime/cache evidence: MXFP4 affine quantized matmul, Metal NA active, mixed-SWA cache, `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=64`, and block-disk `disk_writes=2`.
- Still open: Gemma 12B QAT MXFP4 video/audio, installed-app parity, public tunnel parity, and release signing/notarization.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - Gemma 12B QAT MXFP4 dev-app video/VL proof green

- Ran real Electron dev-app Gemma 4 12B QAT MXFP4 Chat Completions video proof with the 1-second solid-red MP4 fixture.
- Added tracked proof summary `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-video-proof-20260610.json`, `status=pass`.
- Proven: app persisted the `video_url` attachment, server `MEDIA_DIAG` saw one `video_url`, the server decoded the base64 MP4, reported `25 total frames @ 25.0 fps`, extracted `4 frames`, and the assistant answered `The video shows a solid red screen.`
- Runtime/cache evidence: MXFP4 affine quantized matmul, Metal NA active, mixed-SWA cache, `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=65`, and block-disk `disk_writes=2`.
- Still open: Gemma 12B QAT MXFP4 audio, installed-app parity, public tunnel parity, and release signing/notarization.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - Gemma 12B QAT MXFP4 dev-app audio honestly gated

- Ran real Electron dev-app Gemma 4 12B QAT MXFP4 Chat Completions audio proof with a 16 kHz mono WAV saying `audio present`.
- Added tracked proof summary `build/current-real-ui-live-model-gemma4-12b-qat-mxfp4-audio-proof-20260610.json`, `status=fail`.
- Positive evidence before the audio turn: real model loaded as `mllm`, text chat turns completed, mixed-SWA cache remained active, and cache/L2 was green with `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=68`, `l2_tokens_on_disk=68`.
- Red evidence: app attempted an audio turn, server `MEDIA_DIAG` saw `input_audio`, and the API returned `400 - /v1/chat/completions received unsupported media modality audio. Supported modalities: text, vision, video.`
- Boundary: this is an honest capability guard, not a crash or cache failure. Do not claim Gemma 12B QAT MXFP4 audio support in the checkpoint release.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANG_2L dev-app image/VL honestly gated

- Ran real Electron dev-app MiMo V2.5 JANG_2L Chat Completions image proof with `--is-mllm` requested.
- Added tracked proof summary `build/current-real-ui-live-model-mimo-v25-jang2l-image-proof-20260610.json`, `status=fail`.
- Positive evidence before the image turn: 105 GiB artifact loaded, text chat turns completed, affine JANG_2L matmul and Metal NA were active, and MiMo native cache/L2 was green with `mixed_swa_kv_v1`, `mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cached_tokens=39`, `l2_block_tokens_on_disk=110`, and `l2_tokens_on_disk=110`.
- Red evidence: server `MEDIA_DIAG` saw one `image_url`, but the API returned `400 - /v1/chat/completions received unsupported media modality image because the loaded runtime is text-only. Supported modalities: text.`
- Boundary: the bundle has preserved media weights, but runtime metadata marks them `unwired weights_preserved_text_runtime`; do not claim MiMo JANG_2L image/VL support in the checkpoint release.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANG_2L dev-app Responses tools classified red

- Ran real Electron dev-app MiMo V2.5 JANG_2L Responses built-in tool proof with `max_tokens=384`, `max_prompt_tokens=12000`, `max_tool_iterations=8`, and thinking off.
- Added tracked proof summary `build/current-real-ui-live-model-mimo-v25-jang2l-responses-tools-proof-20260610.json`, `status=fail`.
- Positive evidence: first turn used `/v1/responses`, emitted one `run_command` tool call, sent a scoped tool-result follow-up with `previous_response_id`, and completed one tool loop. Runtime/cache was live: peak memory `109374.2 MB`, `cache_hit_tokens=1071`, `cache_detail=paged`, `l2_block_tokens_on_disk=3784`, block-disk `disk_hits=18`, and `disk_writes=60`.
- Red evidence: full two-turn loop did not finish; harness failed with `CDP timeout: Runtime.evaluate` while the second Responses request was still active. No final probe-file semantics were verified.
- Boundary: do not claim MiMo JANG_2L Responses/long-tool-loop support in the checkpoint release. Cache/L2 is not the blocker for this row.
- No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - Installed app runtime parity green after local rebuild

- Ran no-heavy installed-app runtime parity audit before rebuilding: `build/current-installed-app-runtime-parity-audit-after-june10-devapp-proofs-20260610.json`, `status=open`.
- Pre-rebuild blocker was narrow: both bundled Python and packaged source mirrors mismatched current source only for `vmlx_engine/utils/jang_loader.py` (`jang>=2.5.29` in installed app versus `jang>=2.5.30` in current source).
- Rebuilt and locally installed `/Applications/vMLX.app` with `panel/scripts/build-and-install.sh`. This is an ad-hoc/local app-dir install path, not a notarized release DMG.
- Reran audit: `build/current-installed-app-runtime-parity-audit-after-local-install-20260610.json`, `status=pass`, `missing_or_stale=[]`, `installed_bundled_engine_hash_parity=true`, `installed_packaged_engine_source_hash_parity=true`, `serve_help_runs=true`, `xml_function_tool_parser_cli=true`, parser/reasoning settings wired, model-owned generation defaults wired, max-output/max-context settings wired, Responses stream cache-detail metrics wired, and single-model gateway cache endpoint routing wired.
- Verified `/Applications/vMLX.app` with `codesign --verify --deep --strict --verbose=2`; result was valid on disk and satisfies its designated requirement.
- Regenerated no-heavy release manifest as `build/current-release-regression-manifest-after-local-installed-app-parity-20260610.json`; it remains `status=fail`, `prepackage_ready=false`, and `release_ready=false`, but installed/staged app runtime parity components are both green.
- Boundary: installed-app runtime/source parity is green, but this does not clear model-specific installed-app chat proofs, public tunnel parity, DMG package/sign/notarize, tag, upload, public release, or the remaining MiMo/N2/Gemma red rows.

# 2026-06-10 - N2 JANGTQ2 installed-app Responses/tool/cache proof green

- Reduced blocker: `api/ui` plus `cache/storage` for the N2 checkpoint candidate in the rebuilt local `/Applications/vMLX.app`.
- Ran `panel/scripts/live-real-ui-model-proof.mjs` with `VMLINUX_REAL_UI_APP_PATH=/Applications/vMLX.app`, `VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`, `VMLINUX_REAL_UI_WIRE_API=responses`, built-in tools enabled, server cache controls enabled, `--is-mllm`, temperature `0`, top_p `1`, max tokens `128`.
- Proof summary: `build/current-real-ui-installed-app-n2-jangtq2-responses-tools-cache-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-responses-tools-cache-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-responses-tools-cache-20260610-chat.png`.
- Proven: installed app UI launched, real 101 GiB N2 JANGTQ2 loaded as MLLM, `/v1/responses` used, two built-in `run_command` calls executed, tool probe files contained `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`, visible assistant turns completed, renderer content deltas counted `8` and `15`, no raw parser/reasoning leak, and server cache controls were verified in the app settings surface.
- Cache/runtime proof: `hybrid_ssm_v1`, attention-only TurboQuant KV storage boundary, native SSM companion state, async rederive component, `cache_detail=paged+ssm`, `cached_tokens=384`, `l2_block_tokens_on_disk=3582`, `l2_ssm_tokens_on_disk=17086`, `l2_tokens_on_disk=20668`, `block_disk_hits=110`, `block_disk_writes=59`, and `ssm_disk_hits=1`.
- Boundary: this clears N2 JANGTQ2 installed-app default Responses/tool/cache parity only. It does not clear public tunnel SSE parity, N2 audio, N2 JANG_1L Metal OOM, stricter custom prompt quality, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - N2 JANGTQ2 installed-app image/video proof green

- Reduced blocker: `media` plus `api/ui` for the N2 checkpoint candidate in the rebuilt local `/Applications/vMLX.app`.
- Image proof summary: `build/current-real-ui-installed-app-n2-jangtq2-image-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-image-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-image-20260610-chat.png`.
- Video proof summary: `build/current-real-ui-installed-app-n2-jangtq2-video-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-video-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-video-20260610-chat.png`.
- Proven image: installed app UI launched, image attachment persisted, server `MEDIA_DIAG` saw one `image_url`, runtime processed `num_images_processed=1`, and assistant answered `Red`.
- Proven video: installed app UI launched, video attachment persisted, server `MEDIA_DIAG` saw one `video_url`, server decoded base64 MP4, extracted `4` frames from the 25 fps clip, runtime processed `num_images_processed=4`, and assistant answered `The video shows a solid red screen with no visible movement or change.`
- Cache/runtime proof for both: `hybrid_ssm_v1`, attention-only TurboQuant KV, native SSM companion state, `cache_detail=paged+ssm`, `cached_tokens=18`, `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`, `l2_tokens_on_disk=118`, `block_disk_hits=3`, `block_disk_writes=2`, and SSM disk stores `2`.
- Boundary: this clears N2 JANGTQ2 installed-app image/video only. It does not clear N2 audio, N2 JANG_1L Metal OOM, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Gemma 12B MXFP4 installed-app Responses/tool/cache proof green

- Reduced blocker: `api/ui` plus `cache/storage` for the Gemma 12B MXFP4 checkpoint candidate in the rebuilt local `/Applications/vMLX.app`.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-mxfp4-responses-tools-cache-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-responses-tools-cache-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-responses-tools-cache-20260610-chat.png`.
- Proven: installed app UI launched, Gemma 12B QAT MXFP4 loaded as MLLM, `/v1/responses` used, two built-in `run_command` calls executed, tool probe files contained `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`, visible assistant turns completed, renderer content deltas counted `16` and `24`, no raw parser/reasoning leak, and server cache controls were verified in the app settings surface.
- Cache/runtime proof: MXFP4 affine matmul with Metal NA active, active memory about `7772.4 MB`, peak about `10512.9 MB`, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=3538`, `l2_block_tokens_on_disk=3584`, `l2_tokens_on_disk=3584`, `block_disk_hits=30`, and `block_disk_writes=58`.
- Boundary: this clears Gemma 12B MXFP4 installed-app default Responses/tool/cache parity only. It does not clear Gemma installed-app media, Gemma audio, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Gemma 12B MXFP4 installed-app image/VL proof green

- Reduced blocker: `media` plus `api/ui` for Gemma 12B MXFP4 in the rebuilt local `/Applications/vMLX.app`.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-mxfp4-image-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-image-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-image-20260610-chat.png`.
- Proven: installed app UI launched, image attachment persisted, server `MEDIA_DIAG` saw one `image_url`, Gemma media fallback ran with `1 image(s)`, and the assistant answered `Red`; `imageSemanticVerified=true`.
- Cache/runtime proof: MXFP4 affine matmul with Metal NA active, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=70`, `l2_tokens_on_disk=70`, and `block_disk_writes=2`.
- Boundary: this clears Gemma 12B MXFP4 installed-app image only. It does not clear installed-app video, Gemma audio, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Gemma 12B MXFP4 installed-app video/VL proof green

- Reduced blocker: `media` plus `api/ui` for Gemma 12B MXFP4 in the rebuilt local `/Applications/vMLX.app`.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-mxfp4-video-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-video-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-video-20260610-chat.png`.
- Proven: installed app UI launched, video attachment persisted, server `MEDIA_DIAG` saw one `video_url`, server decoded the base64 MP4, reported `25` frames at `25.0 fps`, extracted `4` frames, Gemma media fallback ran, and the assistant answered `The video shows a solid red background with no movement or changes.`
- Cache/runtime proof: MXFP4 affine matmul with Metal NA active, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=70`, `l2_tokens_on_disk=70`, and `block_disk_writes=2`.
- Boundary: this clears Gemma 12B MXFP4 installed-app video only. It does not clear Gemma audio, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - N2 JANGTQ2 installed-app audio honestly gated

- Reduced blocker: `media` plus `api/ui` for the N2 checkpoint candidate in the rebuilt local `/Applications/vMLX.app`.
- Proof summary: `build/current-real-ui-installed-app-n2-jangtq2-audio-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-audio-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-n2-jangtq2-audio-20260610-chat.png`.
- Positive evidence: installed app UI launched, the real 101 GiB N2 JANGTQ2 row loaded as MLLM, two visible text turns completed before audio, the app forced multimodal for one attached audio file, and server `MEDIA_DIAG` saw one `input_audio`.
- Runtime/cache evidence before the audio gate: `hybrid_ssm_v1`, attention-only TurboQuant KV, native SSM companion state, `cache_detail=paged+ssm`, `cached_tokens=18`, `l2_block_tokens_on_disk=50`, `l2_ssm_tokens_on_disk=68`, `l2_tokens_on_disk=118`, `block_disk_hits=3`, `block_disk_writes=2`, and SSM disk stores `2`.
- Red boundary: the API returned `400 - /v1/chat/completions received unsupported media modality audio. Supported modalities: text, vision, video.` This is an honest capability gate, not a load/cache/L2 failure.
- Boundary: this clears installed-app audio classification only. It does not clear N2 audio support, N2 JANG_1L, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Gemma 12B MXFP4 installed-app audio honestly gated

- Reduced blocker: `media` plus `api/ui` for Gemma 12B MXFP4 in the rebuilt local `/Applications/vMLX.app`.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-mxfp4-audio-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-audio-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-mxfp4-audio-20260610-chat.png`.
- Positive evidence: installed app UI launched, the real Gemma 12B QAT MXFP4 row loaded as MLLM, two visible text turns completed before audio, the app forced multimodal for one attached audio file, and server `MEDIA_DIAG` saw one `input_audio`.
- Runtime/cache evidence before the audio gate: MXFP4 affine matmul with Metal NA active, native `mixed_swa_kv_v1`, generic TurboQuant KV correctly disabled, `cache_detail=paged+mixed_swa`, `cached_tokens=20`, `l2_block_tokens_on_disk=70`, `l2_tokens_on_disk=70`, and block-disk writes `2`.
- Red boundary: the API returned `400 - /v1/chat/completions received unsupported media modality audio. Supported modalities: text, vision, video.` This is an honest capability gate, not a load/cache/L2 failure.
- Boundary: this clears installed-app audio classification only. It does not clear Gemma audio support, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - N2 JANG_1L override launch red and MiMo installed-app text/cache green

- Reduced blocker: `runtime/kernel` for Nex/N2 Pro JANG_1L and `api/ui` plus `cache/storage` for MiMo JANG_2L installed-app checkpoint scope.
- N2 JANG_1L no-load preflight: `build/current-n2-pro-jang1l-local-memory-preflight-20260610-after-installed-app-proofs.json`, `status=open`, `decision=do_not_launch`; indexed payload `110.57 GiB`, required available `118.57 GiB`, observed available `112.77 GiB`, gap `5.8 GiB`.
- Eric explicitly overrode the JANG_1L launch-safe gate. Live override artifact `build/current-n2-jang1l-live-chat-cache-override-20260610.json` is `status=fail`, `phase=server_startup`; server log is `build/current-n2-jang1l-live-chat-cache-override-20260610.server.log`.
- N2 JANG_1L override details: launched one-at-a-time with `max_tokens=16`, prefill batch `64`, prefill step `128`, completion batch `32`, SSM cache `128 MB`, paged cache block size `64`, max blocks `256`, and block L2 `2 GB`. It still aborted before health with `[METAL] Command buffer execution failed: Insufficient Memory` after `Wired limit set to 115 GB (model 119 GB)`.
- MiMo installed-app proof summary: `build/current-real-ui-installed-app-mimo-v25-jang2l-text-cache-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-text-cache-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-text-cache-20260610-chat.png`.
- MiMo proven: local rebuilt `/Applications/vMLX.app` launched, real 105 GiB MiMo JANG_2L loaded, Chat Completions produced exact visible `MIMO_INSTALLED_TEXT_ONE` and `MIMO_INSTALLED_TEXT_TWO`, server cache controls were visible, no parser/reasoning leak was recorded, and generation defaults were applied.
- MiMo runtime/cache evidence: active memory `105017.5 MB`, peak `105961.1 MB`, Metal NA active affine JANG_2L matmul, single-active decode, native `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cached_tokens=41`, `l2_block_tokens_on_disk=114`, `l2_tokens_on_disk=114`, and block-disk writes `3`.
- Boundary: MiMo installed-app text/cache is green, but installed-app tools/media/JANGTQ_2 exactness/speed remain open. N2 JANG_1L remains red until a lower-peak runtime strategy exists. No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANG_2L installed-app tool loop classified red

- Reduced blocker: `api/ui` plus `parser/template` for MiMo JANG_2L installed-app built-in tool loop.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jang2l-tools-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-tools-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-tools-20260610-chat.png`.
- Positive evidence: local rebuilt `/Applications/vMLX.app` launched, the real 105 GiB MiMo JANG_2L row loaded, built-in tool events streamed, one `run_command` executed on the second turn, and `real_ui_tool_probe_2.txt` contained `REAL_UI_LIVE_TOOL_TWO`.
- Red evidence: the release assertion failed because `long_tool_loop` was not recorded. The first requested marker mutated to `REAL_UI_LAND_TOOL_ONE`, the expected first probe file was not created, and first visible content degraded into repetitive tool-planning prose.
- Runtime/cache evidence: active memory `105384.7 MB`, peak `109903.3 MB`, native `mixed_swa_kv_v1` with `mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=4552`, last cached tokens `3481`, and `l2_block_tokens_on_disk=4720`.
- Boundary: MiMo installed-app tools remain red. Cache/L2 is not the blocker. No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANG_2L installed-app image honestly gated

- Reduced blocker: `media` plus `api/ui` for MiMo JANG_2L installed-app image/VL.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jang2l-image-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-image-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jang2l-image-20260610-chat.png`.
- Positive evidence: local rebuilt `/Applications/vMLX.app` launched, real 105 GiB MiMo JANG_2L loaded, two visible text turns completed before media, the app attached one image, and server `MEDIA_DIAG` saw one `image_url`.
- Red boundary: the server returned `400 - /v1/chat/completions received unsupported media modality image because the loaded runtime is text-only. Supported modalities: text.` Server log records that forced MLLM was overridden because MiMo media weights are preserved but `unwired weights_preserved_text_runtime`.
- Runtime/cache evidence: active memory `105016.1 MB`, peak `105842.8 MB`, native `mixed_swa_kv_v1` with `mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cached_tokens=39`, `l2_block_tokens_on_disk=110`, and block-disk writes `3`.
- Boundary: MiMo installed-app media remains unsupported by honest guard. This does not clear tools, media, JANGTQ_2 exactness, package/sign/notarize/tag/upload, or release readiness.

# 2026-06-10 - N2 JANG_1L launch-safe gate refreshed

- Reduced blocker: `runtime/kernel` plus `cache/storage` scheduling proof for Nex/N2 Pro JANG_1L.
- No-load preflight: `build/current-n2-pro-jang1l-local-memory-preflight-launch-safe-20260610.json`, `status=open`, `decision=do_not_launch`; indexed payload `110.57 GiB`, required available `118.57 GiB`, observed available `114.23 GiB`, gap `4.34 GiB`.
- Launch-safe chat/cache gate: `build/current-n2-jang1l-chat-cache-launch-safe-20260610.json`, `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`; observed available `114.22 GiB`, required available `118.57 GiB`, gap `4.35 GiB`.
- Requested probes were preserved in the gate artifact: tool, Responses, Responses stream, and L2 restart. The model was not launched because the safe gate blocked before `Popen`.
- Boundary: this is current safe scheduling evidence only. The previous explicit override OOM remains the live failure evidence; JANG_1L still needs a lower-peak runtime strategy before any support claim. No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - MiMo JANGTQ_2 installed-app short text/cache green

- Reduced blocker: `api/ui` plus `cache/storage` for MiMo JANGTQ_2 installed-app short exact text/cache.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jangtq2-text-cache-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-text-cache-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-text-cache-20260610-chat.png`.
- Proven: local rebuilt `/Applications/vMLX.app` launched, real 79 GiB MiMo JANGTQ_2 loaded, Chat Completions produced exact visible turns `MIMO_JANGTQ2_TEXT_ONE` and `MIMO_JANGTQ2_TEXT_TWO`, server cache controls were visible, no parser/reasoning leak was recorded, and generation defaults were applied.
- Runtime/cache evidence: active memory `76484.8 MB`, peak `77037.2 MB`, native TurboQuant codebook routed experts, `mixed_swa_kv_v1` with `mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=42`, `l2_block_tokens_on_disk=120`, `l2_tokens_on_disk=120`, and block-disk writes `3`.
- Boundary: this clears only short installed-app text/cache for MiMo JANGTQ_2. It does not clear older broader JANGTQ_2 artifact exactness failures, tools, media, source-vs-quant, public tunnel, package/sign/notarize/tag/upload, or release readiness.

# 2026-06-10 - MiMo JANGTQ_2 installed-app image honestly gated

- Reduced blocker: `media` plus `api/ui` for MiMo JANGTQ_2 installed-app image/VL.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jangtq2-image-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-image-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-image-20260610-chat.png`.
- Positive evidence: local rebuilt `/Applications/vMLX.app` launched, real 79 GiB MiMo JANGTQ_2 loaded, two visible text turns completed before media, the app attached one image, and server `MEDIA_DIAG` saw one `image_url`.
- Red boundary: the server returned `400 - /v1/chat/completions received unsupported media modality image because the loaded runtime is text-only. Supported modalities: text.` Server log records that forced MLLM was overridden because MiMo media weights are preserved but `unwired weights_preserved_text_runtime`.
- Runtime/cache evidence before the gate: active memory `76491.8 MB`, peak `77127.2 MB`, native `mixed_swa_kv_v1` with `mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=39`, `l2_block_tokens_on_disk=132`, `l2_tokens_on_disk=132`, and block-disk writes `3`.
- Boundary: MiMo JANGTQ_2 installed-app media remains unsupported by honest guard. Vision tensors/config metadata do not equal runtime media support. No package/sign/notarize/tag/upload/release action was run.

# 2026-06-10 - Gemma 12B JANG4M installed-app Responses/tool/cache green

- Reduced blocker: `api/ui` plus `cache/storage` for Gemma 12B JANG4M installed-app parity.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-jang4m-responses-tools-cache-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-responses-tools-cache-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-responses-tools-cache-20260610-chat.png`.
- Proven: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` loaded as MLLM, `/v1/responses` used, two built-in `run_command` calls executed, tool-result continuations used `previous_response_id`, visible assistant turns completed, content/tool deltas streamed, and no parser/reasoning leak was recorded.
- Runtime/cache evidence: JANG affine matmul with Metal NA active, active memory `9889.4 MB`, peak `12630.4 MB`, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=3538`, `l2_block_tokens_on_disk=3571`, `l2_tokens_on_disk=3571`, block-disk hits `30`, and block-disk writes `58`.
- Boundary: this clears Gemma 12B JANG4M installed-app Responses/tool/cache only. It does not clear installed-app JANG4M image/video/audio, larger Gemma QAT rows, public tunnel SSE, package/sign/notarize/tag/upload, or release readiness.

# 2026-06-10 - Gemma 12B JANG4M installed-app image/VL proof green

- Reduced blocker: `media` plus `api/ui` for Gemma 12B JANG4M installed-app image/VL parity.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-jang4m-image-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-image-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-image-20260610-chat.png`.
- Proven: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` loaded as MLLM, two visible text turns completed before media, the app attached one red PNG image, Gemma media fallback ran with `1 image(s)`, and the assistant answered `Red`; `imageSemanticVerified=true`.
- Runtime/cache evidence: active memory `9892.5 MB`, peak `10450.3 MB`, JANG affine matmul with Metal NA active, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=77`, `l2_tokens_on_disk=77`, and block-disk writes `2`.
- Boundary: this clears Gemma 12B JANG4M installed-app image only. It does not clear installed-app JANG4M video/audio, larger Gemma QAT rows, public tunnel SSE, package/sign/notarize/tag/upload, or release readiness.

# 2026-06-10 - N2 JANG_1L high-free launch still Metal OOM

- Reduced blocker: `runtime/kernel` for Nex/N2 Pro JANG_1L on the 128 GiB host.
- Fresh no-load preflight: `build/current-n2-pro-jang1l-local-memory-preflight-ultrafree-20260610.json`, `status=open`, `decision=do_not_launch`; indexed payload `110.57 GiB`, required available `118.57 GiB`, observed available `114.09 GiB`, gap `4.48 GiB`.
- Per Eric's launch instruction, ran a real live gate anyway with lowered JANG_1L headroom (`3 GiB`) and one-at-a-time low-peak knobs: prefill batch `64`, prefill step `128`, completion batch `32`, SSM state cache `128 MB`, paged cache block size `64`, max cache blocks `256`, block L2 `2 GB`, max output `16`, server max tokens `256`.
- Live proof: `build/current-n2-jang1l-live-chat-cache-ultrafree-20260610.json`, `status=fail`, `phase=server_startup`; server log `build/current-n2-jang1l-live-chat-cache-ultrafree-20260610.server.log`.
- Runtime evidence before abort: server detected `family=qwen3_5_moe`, `tool_parser=qwen`, `reasoning_parser=qwen3`, `cache_type=hybrid`; enabled attention-only TurboQuant KV for the hybrid model while preserving native full-precision SSM/GatedDelta companion state; selected JANG text-only route and mmap JANG loader; patched `482` quant-shape modules; started loading `123` shards; enabled bfloat16 for `512` experts; set `Wired limit set to 115 GB (model 119 GB)`.
- Red boundary: startup aborted before health with `[METAL] Command buffer execution failed: Insufficient Memory`. N2 JANG_1L remains not release-clear and needs an actual lower-peak loader/runtime strategy, not another threshold reduction. N2 JANGTQ2 proof still does not clear JANG_1L.

# 2026-06-10 - Gemma 12B JANG4M installed-app video/VL proof green

- Reduced blocker: `media` plus `api/ui` for Gemma 12B JANG4M installed-app video/VL parity.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-jang4m-video-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-video-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-video-20260610-chat.png`.
- Proven: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` loaded as MLLM with `max_prompt_tokens=12000`, two visible text turns completed before media, the app attached one MP4, server `MEDIA_DIAG` saw `video_url`, server decoded base64 MP4, extracted `4` frames from a 25 fps clip, and the assistant answered `The video shows a solid, static red screen with no movement or changes.`
- Runtime/cache evidence: active memory `9890 MB`, peak `10430.4 MB`, JANG affine matmul with Metal NA active, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, `l2_block_tokens_on_disk=77`, `l2_tokens_on_disk=77`, and block-disk writes `2`.
- Boundary: this clears Gemma 12B JANG4M installed-app video only at the explicit `12000` prompt cap. It does not clear default-4k video, installed-app JANG4M audio, larger Gemma QAT rows, public tunnel SSE, package/sign/notarize/tag/upload, or release readiness.

# 2026-06-10 - Gemma 12B JANG4M installed-app audio classified red

- Reduced blocker: `media` plus `api/ui` for Gemma 12B JANG4M installed-app audio.
- Proof summary: `build/current-real-ui-installed-app-gemma4-12b-jang4m-audio-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-audio-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-12b-jang4m-audio-20260610-chat.png`.
- Positive evidence: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` loaded as MLLM, two visible text turns completed before audio, the app attached one WAV, `persistedAudioAttachment=true`, server `MEDIA_DIAG` saw `input_audio`, and server decoded the base64 WAV.
- Red evidence: final audio turn ended with empty visible assistant content, `visibleAssistantTurnsComplete=false`, `audioSemanticVerified=false`, and no `audio_where_supported` proof surface was recorded.
- Runtime/cache evidence before the audio failure: active memory `9890 MB`, JANG affine matmul with Metal NA active, native `mixed_swa_kv_v1`, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=20`, and block L2 writes before the audio/media turn. Media cache correctly skipped text-only prefix/paged cache store for the path-dependent audio request.
- Boundary: this classifies JANG4M installed-app audio as red. It does not invalidate installed-app JANG4M text/tools/image/video green rows, and it does not clear public tunnel SSE, package/sign/notarize/tag/upload, or release readiness.

# 2026-06-10 - MiMo JANGTQ_2 installed-app tool loop green

- Reduced blocker: `api/ui` plus `parser/template` for MiMo V2.5 JANGTQ_2 installed-app built-in tool loop.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jangtq2-tools-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-tools-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-tools-20260610-chat.png`.
- Proven: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded, Chat Completions used built-in `run_command`, the two-turn tool loop completed, `real_ui_tool_probe_1.txt` contained `REAL_UI_LIVE_TOOL_ONE`, `real_ui_tool_probe_2.txt` contained `REAL_UI_LIVE_TOOL_TWO`, visible assistant turns completed, and no raw parser/reasoning leak was recorded.
- Runtime/cache evidence: active memory `76763.1 MB`, peak `81328.7 MB`, single-active decode, native TurboQuant codebook routed experts (`profile=JANGTQ_2`), prestacked routed layout, `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=4548`, `l2_block_tokens_on_disk=4225`, `l2_tokens_on_disk=4225`, block-disk hits `36`, and block-disk writes `68`.
- Boundary: this clears MiMo JANGTQ_2 installed-app Chat Completions default built-in tool loop only. It does not clear broader JANGTQ_2 exactness/source-vs-quant rows, Responses tool path, media, package/sign/notarize/tag/upload, or release readiness. It also does not clear MiMo JANG_2L tools, which remain separately red.

# 2026-06-10 - MiMo JANGTQ_2 installed-app Responses tool loop green

- Reduced blocker: `api/ui` plus `responses` for MiMo V2.5 JANGTQ_2 installed-app built-in tool loop.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jangtq2-responses-tools-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-responses-tools-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-responses-tools-20260610-chat.png`.
- Proven: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded, `/v1/responses` used, built-in `run_command` executed, tool-result follow-ups used `previous_response_id` with `function_call_output`, `responses_delta_streaming` and `responses_cache_detail_usage` surfaces were recorded, exact probe files were created, visible assistant turns completed, and no raw parser/reasoning leak was recorded.
- Runtime/cache evidence: active memory `76763.1 MB`, peak `81328.7 MB`, single-active decode, TurboQuant codebook routed experts (`profile=JANGTQ_2`), prestacked routed layout, `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=4548`, `l2_block_tokens_on_disk=4225`, `l2_tokens_on_disk=4225`, block-disk hits `36`, and block-disk writes `68`.
- Boundary: this clears MiMo JANGTQ_2 installed-app Responses default built-in tool loop only. It does not clear broader JANGTQ_2 literal/JSON/source-vs-quant exactness rows, media, package/sign/notarize/tag/upload, or release readiness. It also does not clear MiMo JANG_2L tools, which remain separately red.

# 2026-06-10 - MiMo JANGTQ_2 installed-app exact output still red

- Reduced blocker: `decode/artifact exactness` for MiMo V2.5 JANGTQ_2 installed-app literal/JSON output.
- Proof summary: `build/current-real-ui-installed-app-mimo-v25-jangtq2-exact-output-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-exact-output-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-installed-app-mimo-v25-jangtq2-exact-output-20260610-chat.png`.
- Positive evidence: local rebuilt `/Applications/vMLX.app` launched, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded, Chat Completions streamed visible assistant turns, no raw parser/reasoning leak was recorded, no tools or reasoning were persisted, server cache controls were visible, and generation defaults were applied.
- Runtime/cache evidence: active memory `76483.5 MB`, peak `77024.8 MB`, single-active decode, TurboQuant codebook routed experts (`profile=JANGTQ_2`), prestacked routed layout, native `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=41`, `l2_block_tokens_on_disk=117`, `l2_tokens_on_disk=117`, and block-disk writes `3`.
- Red evidence: the exact text probe expected `ACK-CB-742` but returned `ACKCB-742`; the exact JSON probe expected `{"status":"ok","value":"blue-cat"}` but returned only `{"`. This confirms MiMo JANGTQ_2 installed-app exactness remains red and should be investigated in artifact/logit/quant/decode contract, not parser repair or cache disabling.

# 2026-06-10 - N2 JANG_1L after-MiMo launch-safe refresh still skipped

- Reduced blocker: `runtime/kernel` plus careful-RAM scheduling for Nex/N2 Pro JANG_1L.
- Fresh no-load preflight: `build/current-n2-pro-jang1l-local-memory-preflight-after-mimo-exact-20260610.json`, `status=open`, `decision=do_not_launch`; indexed payload `110.57 GiB`, required available `118.57 GiB`, observed available `113.29 GiB`, gap `5.28 GiB`.
- Launch-safe chat/cache gate: `build/current-n2-jang1l-chat-cache-after-mimo-exact-20260610.json`, `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`; observed available `113.28 GiB`, required available `118.57 GiB`, gap `5.29 GiB`.
- Requested probes were preserved in the gate artifact: tool, Responses, Responses stream, and L2 restart. No weights were launched and the cache directory stayed empty.
- Boundary: this is current no-load/scheduling evidence only. It does not prove N2 JANG_1L runtime/cache/API/UI and does not invalidate N2 JANGTQ2 as the current N2 checkpoint candidate. The earlier forced below-gate launch remains the live Metal OOM evidence.

# 2026-06-10 - MiMo JANGTQ_2 dev-app Responses tool/cache green

- Reduced blocker: `api/ui` plus `responses` plus `cache/storage` for current Electron dev-build MiMo V2.5 JANGTQ_2.
- Proof summary: `build/current-real-ui-dev-app-mimo-v25-jangtq2-responses-tools-cache-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-responses-tools-cache-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-responses-tools-cache-20260610-chat.png`.
- Proven: `npm run dev` Electron app launched (`uiLaunchMode=electron-dev`), real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded, `/v1/responses` used, built-in `run_command` executed in both turns, `previous_response_id` tool follow-ups with `function_call_output` were used, Responses delta/cache-detail surfaces were recorded, and exact probe files were created: `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`.
- Runtime/cache evidence: active memory `76763.1 MB`, peak `81328.7 MB`, single-active decode, TurboQuant codebook routed experts (`profile=JANGTQ_2`), prestacked routed layout, native `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=4548`, `l2_block_tokens_on_disk=4225`, `l2_tokens_on_disk=4225`, block-disk hits `36`, and block-disk writes `68`.
- Boundary: this clears MiMo JANGTQ_2 current dev-build Responses/tool/cache parity only. It does not clear broader JANGTQ_2 literal/JSON/source-vs-quant exactness or media support.

# 2026-06-10 - MiMo JANGTQ_2 dev-app exact output still red

- Reduced blocker: `decode/artifact exactness` for current Electron dev-build MiMo V2.5 JANGTQ_2 literal/JSON output.
- Proof summary: `build/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-exact-output-20260610-chat.png`.
- Positive evidence: `npm run dev` Electron app launched (`uiLaunchMode=electron-dev`), real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded, Chat Completions streamed visible assistant turns, no raw parser/reasoning leak was recorded, no tools or reasoning were persisted, server cache controls were visible, and generation defaults were applied.
- Runtime/cache evidence: active memory `76483.5 MB`, peak `77024.8 MB`, single-active decode, TurboQuant codebook routed experts (`profile=JANGTQ_2`), prestacked routed layout, native `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=41`, `l2_block_tokens_on_disk=117`, `l2_tokens_on_disk=117`, and block-disk writes `3`.
- Red evidence: the exact text probe expected `ACK-CB-742` but returned `ACKCB-742`; the exact JSON probe expected `{"status":"ok","value":"blue-cat"}` but returned only `{"`. This matches the installed-app exactness failure and keeps MiMo JANGTQ_2 exactness assigned to artifact/logit/quant/decode contract.

# 2026-06-10 - N2 JANGTQ2 dev-app exact output green

- Reduced blocker: `api/ui` plus `decode exactness` plus `cache/storage` for current Electron dev-build Nex/N2 Pro JANGTQ2.
- Proof summary: `build/current-real-ui-dev-app-n2-jangtq2-exact-output-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-dev-app-n2-jangtq2-exact-output-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-dev-app-n2-jangtq2-exact-output-20260610-chat.png`.
- Proven: `npm run dev` Electron app launched (`uiLaunchMode=electron-dev`), real `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2` loaded, Chat Completions returned exact text `N2-ACK-742` and exact JSON `{"status":"ok","value":"n2-blue"}`, no raw parser/reasoning leak was recorded, no tools or reasoning were persisted, server cache controls were visible, and generation defaults were applied.
- Runtime/cache evidence: active memory `103805 MB`, peak `104441.4 MB`, model type `mllm`, TurboQuant codebook `weight_format=mxtq`, `profile=JANGTQ2`, prestacked routed layout, native `hybrid_ssm_v1` with `attention_kv`, `ssm_companion_state`, and `async_rederive`, attention-only TurboQuant KV enabled, native SSM companion state preserved, `cache_detail=paged+ssm`, `cache_hit_tokens=21`, `l2_block_tokens_on_disk=59`, `l2_ssm_tokens_on_disk=80`, `l2_tokens_on_disk=139`, block-disk hits `3`, block-disk writes `2`, and SSM companion stores `2`.
- Boundary: this clears N2 JANGTQ2 current dev-build exact text/JSON output only. It does not clear N2 JANG_1L, audio, public tunnel SSE parity, stricter custom long-delta prompt quality, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Gemma 12B MXFP4 dev-app exact output green

- Reduced blocker: `api/ui` plus `decode exactness` plus `cache/storage` for current Electron dev-build Gemma 12B QAT MXFP4.
- Proof summary: `build/current-real-ui-dev-app-gemma4-12b-mxfp4-exact-output-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-12b-mxfp4-exact-output-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-12b-mxfp4-exact-output-20260610-chat.png`.
- Proven: `npm run dev` Electron app launched (`uiLaunchMode=electron-dev`), real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4` loaded, Chat Completions returned exact text `GEMMA-ACK-742` and exact JSON `{"status":"ok","value":"gemma-blue"}`, no raw parser/reasoning leak was recorded, no tools or reasoning were persisted, server cache controls were visible, and generation defaults were applied.
- Runtime/cache evidence: active memory `7558.3 MB`, peak `7887 MB`, `weight_format=mxfp4`, `profile=MXFP4`, `weight_matmul_dispatch=mlx_affine_quantized_matmul`, Metal NA active, native `mixed_swa_kv_v1`, generic TurboQuant KV correctly disabled for rotating mixed-SWA metadata, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=22`, `l2_block_tokens_on_disk=61`, `l2_tokens_on_disk=61`, and block-disk writes `2`.
- Boundary: this clears Gemma 12B QAT MXFP4 current dev-build exact text/JSON output only. It does not clear Gemma audio, larger Gemma QAT rows, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Gemma 12B JANG4M dev-app exact output green

- Reduced blocker: `api/ui` plus `decode exactness` plus `cache/storage` for current Electron dev-build Gemma 12B JANG4M.
- Proof summary: `build/current-real-ui-dev-app-gemma4-12b-jang4m-exact-output-proof-20260610.json`, `status=pass`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-12b-jang4m-exact-output-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-dev-app-gemma4-12b-jang4m-exact-output-20260610-chat.png`.
- Proven: `npm run dev` Electron app launched (`uiLaunchMode=electron-dev`), real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M` loaded, Chat Completions returned exact text `JANG4M-ACK-742` and exact JSON `{"status":"ok","value":"jang4m-blue"}`, no raw parser/reasoning leak was recorded, no tools or reasoning were persisted, Gemma4 tool/reasoning parsers auto-detected, server cache controls were visible, and generation defaults were applied.
- Runtime/cache evidence: active memory `9680.4 MB`, peak `9950.7 MB`, `weight_format=jang_affine`, `profile=JANG_4M`, `weight_matmul_dispatch=mlx_affine_quantized_matmul`, Metal NA active, native `mixed_swa_kv_v1`, generic TurboQuant KV correctly disabled for rotating mixed-SWA metadata, `cache_detail=paged+mixed_swa`, `cache_hit_tokens=24`, `l2_block_tokens_on_disk=66`, `l2_tokens_on_disk=66`, and block-disk writes `2`.
- Boundary: this clears Gemma 12B JANG4M current dev-build exact text/JSON output only. It does not clear Gemma audio, larger Gemma QAT rows, public tunnel SSE parity, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - N2 JANG_1L deferred startup eval partial fix

- Reduced blocker: `runtime/kernel` startup for Nex/N2 Pro JANG_1L on the 128 GiB host.
- Code fix: `vmlx_engine/utils/tokenizer.py` now detects qwen3_5_moe affine `JANG_1L` and passes `skip_eval=True` to `load_jang_model`, avoiding eager all-parameter startup eval for this 397B-class row. Focused policy tests passed and verify JANGTQ/non-N2 rows do not inherit the policy.
- Proof summary: `build/current-n2-jang1l-deferred-eval-startup-proof-20260610.json`, `status=open`; preflight `build/current-n2-pro-jang1l-local-memory-preflight-deferred-eval-live-attempt-20260610.json` scheduled live proof with `available_gib=114.04`, `required_available_gib=113.57`; live artifact `build/current-n2-jang1l-live-chat-cache-deferred-eval-live-attempt-20260610.json` is `status=fail` but no longer fails at startup.
- Proven: real JANG_1L loaded, reached `/health`, recorded qwen3_5_moe/qwen/qwen3/hybrid detection, loaded `123` shards with `482` quant-shape repairs, initialized attention TurboQuant KV plus native SSM companion, paged cache, block L2, and SSM companion L2, and completed one bounded Chat Completions request with HTTP `200`.
- Red evidence: after the first bounded request, active Metal working set hit `102%` of the `107.5GB` cap; cache warm/hit, tool, Responses, Responses stream, and full L2 restart probes did not pass. A second experiment with `VMLINUX_METAL_WS_REJECT_PCT=104` plus wired-limit setup crashed on the first request with `[METAL] Command buffer execution failed: Insufficient Memory`, so raising/bypassing the guard is not an acceptable release fix.
- Boundary: N2 JANG_1L has a real startup/first-chat improvement, but it is not release-clear for cache reuse, tools, Responses, L2 restart, UI, or media. N2 JANGTQ2 remains the checkpoint N2 row.

# 2026-06-10 - MiMo JANGTQ_2 dev-app image/VL red

- Reduced blocker: `media` plus `api/ui` for current Electron dev-build MiMo V2.5 JANGTQ_2.
- Proof summary: `build/current-real-ui-dev-app-mimo-v25-jangtq2-image-proof-20260610.json`, `status=fail`; raw proof and screenshot are `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-image-proof-20260610-proof.json` and `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-image-proof-20260610-chat.png`.
- Positive evidence: `npm run dev` Electron app launched, real `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` loaded, two visible text turns completed before media, one image attachment was sent, server `MEDIA_DIAG` saw `image_url`, no raw parser/reasoning leak was recorded, server cache controls were visible, and generation defaults were applied.
- Runtime/cache evidence: active memory `76491.8 MB`, peak `77127.2 MB`, TurboQuant codebook routed experts (`profile=JANGTQ_2`), prestacked routed layout, native `mixed_swa_kv_v1` with `cache_subtype=mimo_v2_asymmetric_swa`, `cache_detail=paged`, `cache_hit_tokens=39`, `l2_block_tokens_on_disk=132`, `l2_tokens_on_disk=132`, and block-disk writes `3`.
- Red evidence: image turn returned HTTP `400`: `/v1/chat/completions received unsupported media modality image because the loaded runtime is text-only. Supported modalities: text.` The server log records the reason: MiMo V2 preserved media weights override forced MLLM because bundle metadata marks vision/audio as `unwired weights_preserved_text_runtime`.
- Boundary: this classifies MiMo JANGTQ_2 dev-app image/VL as red. It does not invalidate the MiMo JANGTQ_2 dev-app Responses/tool/cache green row and does not clear MiMo media, exactness, package/sign/notarize/tag/upload, or full `release_ready`.

# 2026-06-10 - Checkpoint DMGs built, signed, notarized, stapled, verified

- Built checkpoint-scope vMLX `1.5.56` Sequoia and Tahoe DMGs from the active Python/Electron worktree with the documented override path, not the production release gate:
  `VMLINUX_CHECKPOINT_RELEASE_OVERRIDE=1 VMLX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-checkpoint-dmg-override-after-n2-consumed-20260610.json panel/scripts/build-release-dmgs.sh all`.
- Release manifest produced during the build remains red: `build/current-release-regression-manifest-checkpoint-dmg-override-after-n2-consumed-20260610.json` has `status=fail`, `prepackage_ready=false`, and `release_ready=false`. This is a signed checkpoint artifact for user testing, not a production-clear release.
- Keychain/signing path followed `/Users/eric/wiki/infra/apple-notarization.md`: `~/Library/Keychains/vmlx-build.keychain-db`, Developer ID `Developer ID Application: ShieldStack LLC (55KGF2S5AY)`, and notary profile `vmlx-notary`.
- Build proof: both bundled apps passed critical import/parity checks before DMG creation, including local `vmlx_engine 1.5.56`, bundled local `jang 2.5.30`, Gemma4 unified runtime, MiMo registration, N2/Qwen VLM/runtime imports, JANG/JANGTQ loaders, TurboQuant kernels, audio/VLM dependencies, and source/bundled critical-file parity.
- Notarization command passed:
  `VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh`.
- Verification command passed:
  `panel/scripts/verify-release-dmgs.sh`.
- Final artifacts:
  `panel/release/vMLX-1.5.56-sequoia-arm64.dmg` (`sha256=42c053cd2422e72ef74753cbc240a68a319d6c10ff60c105d5ed4c4c34f34a9c`) and
  `panel/release/vMLX-1.5.56-tahoe-arm64.dmg` (`sha256=b35e6cb55ca0f7e50a9a4a8733f111ea3df070ccf32caa87510c8027f16fb2f2`).
- Apple notary IDs: Sequoia `d29a3974-4674-4812-8fa2-5a7e0da69269`; Tahoe `27ff0109-e023-469d-a634-5c410f37ac3c`.
- Verified status for both DMGs: `hdiutil verify` valid, `codesign --verify` valid, Developer ID authority `ShieldStack LLC (55KGF2S5AY)`, notarization ticket stapled, `xcrun stapler validate` worked, and Gatekeeper `spctl` accepted with `source=Notarized Developer ID`.
- Boundary: no GitHub release, tag, latest manifest upload, public asset upload, or PyPI publish was performed in this lane. Checkpoint claims must stay limited to currently green rows: N2 JANGTQ2, Gemma 12B MXFP4/JANG4M text/tools/cache plus proven image/video rows where recorded, Gemma 26B/31B JANG4M dev image/video rows, and MiMo JANGTQ_2 tool/cache transport. N2 JANG_1L, MiMo exactness/media, DSV4, public tunnel SSE parity, audio support, and full `release_ready` remain open.

# 2026-06-10 - Public checkpoint release surface patched

- Replaced the public `v1.5.56` checkpoint assets on both GitHub release repos:
  `jjang-ai/vmlx` and `jjang-ai/mlxstudio`.
- Current GitHub release asset digests now match the notarized local checkpoint DMGs:
  Sequoia `sha256=42c053cd2422e72ef74753cbc240a68a319d6c10ff60c105d5ed4c4c34f34a9c`,
  Tahoe `sha256=b35e6cb55ca0f7e50a9a4a8733f111ea3df070ccf32caa87510c8027f16fb2f2`,
  Sequoia blockmap `sha256=49f2cc135d52f2713cfb84d5bcfe0e6bd1becc7ad947a43ce4c896820be020f0`,
  Tahoe blockmap `sha256=8ae7d7afd515f793ef639e3e73477329d701295e01c1feada04a3d92d9ad8546`.
- Updated both GitHub release bodies with the same checkpoint SHA text.
- Updated and pushed `latest.json`:
  `jjang-ai/mlxstudio@d69d2d0` on `main`, and `jjang-ai/vmlx@8604edca` on both `main` and `codex/pr-intake-manifest`.
- Force-moved the `jjang-ai/vmlx` `v1.5.56` source tag to `8604edca`, matching current `origin/main`.
- Patched the live `mlx.studio` web root directly:
  `/var/www/mlx.studio/update/latest.json` now matches committed `mlxstudio/latest.json`,
  `/var/www/mlx.studio/download/index.html` now displays the checkpoint Sequoia/Tahoe SHA strings,
  and Cloudflare purge for `mlx.studio` succeeded.
- Public verification after purge:
  `https://mlx.studio/update/latest.json` returns HTTP 200 with `cache-control: no-cache, no-store, must-revalidate`, `cf-cache-status: DYNAMIC`, and the checkpoint SHA values;
  `https://mlx.studio/download/` displays the checkpoint SHA values;
  PyPI reports `vmlx==1.5.56` and `jang==2.5.31`; no `jangtools` or `jang-tools` distribution exists.
- Release-surface proof artifact:
  `build/current-release-surface-contract-after-checkpoint-upload-20260610.json`.
  It remains `status=fail` only because `https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json` is still serving the pre-replacement hashes from GitHub raw CDN cache even though GitHub Contents API, `github.com/.../raw/refs/heads/main/latest.json`, jsDelivr, and the live site updater all show the checkpoint hashes.
- Boundary: this is a public signed/notarized checkpoint release surface, not production-clear runtime release readiness. The manifest still records `release_ready=false` elsewhere for N2 JANG_1L, MiMo exactness/media, DSV4, public tunnel SSE parity, audio support, and full matrix gaps.

# 2026-06-10 - Gemma audio modality source boundary pinned

- Reduced blocker: `model autodetect / capability detection` for Gemma JANG/MXFP/QAT audio honesty.
- Source proof: `build/current-gemma-audio-modality-source-boundary-20260610.json`, plus inventory refresh `build/current-gemma-native-mxfp4-inventory-gate-refresh-20260610.json`.
- Proven in current source: `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`, `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-MXFP4`, `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M`, and `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` all return `audio_declared=false` and runtime modalities `["text","vision","video"]`.
- Artifact facts: the checked 12B JANG4M/MXFP4 bundles have `audio_config` plus `embed_audio.embedding_projection.weight` only, with `audio_tower_weight_count=0`; the checked 26B/31B JANG4M bundles have no audio tower weights. Current source therefore does not infer audio from config/token/projection-only metadata.
- Test proof: `.venv/bin/python -m pytest -q tests/test_engine_audit.py -k 'gemma4_runtime_modalities or gemma4_unified' tests/test_gemma_qat_native_mxfp4_inventory_gate.py` passed (`16 passed`). The Gemma Unified loader test now explicitly simulates missing source runtime before expecting text-loader fallback, keeping it consistent with source-runtime-available tests.
- Boundary: installed-app audio rows that reached an empty answer remain red/stale until rebuilt from current source and reproven. This does not prove audio generation for 12B/26B/31B, and video still requires live frame-through-vision evidence.
- Parallel-agent note: only advertise Gemma audio when `audio_tower.*` weights exist and live E2E audio proof passes. For E2B/E4B bundles that do have audio tower weights, run real audio E2E separately; do not transfer that claim to 12B/26B/31B projection-only/no-audio bundles.

# 2026-06-10 - Qwen empty-args source boundary refreshed

- Reduced blocker: #192/#190 subcase for Qwen/Qwen3.6 Responses tool calls collapsing to `arguments:{}` after a visible preamble.
- Source proof: `build/current-qwen-empty-args-source-boundary-refresh-20260610.json`.
- Verification: `.venv/bin/python -m pytest -q tests/test_server.py -k 'streaming_responses_tool_call_arguments_survive_buffering or streaming_responses_reasoning_tool_call_keeps_arguments or streaming_responses_tool_call_uses_next_output_index_without_text or streaming_responses_required_empty_xml_tool_call_is_rejected or streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments'` passed (`5 passed`).
- Proven: current source schema-filters parsed tool calls with missing required args; the exact preamble plus empty XML `exec_command` shape emits no function_call item, no `response.function_call_arguments.*` event, and no executable `"arguments":"{}"` payload. Required tool mode fails closed with `tool_calls_required`.
- Boundary: do not claim deployed/public tunnel fixed from this. The known Qwen35 public tunnel blocker is still same-model raw SSE recapture/output-index freshness, not current-source empty-args parsing. Do not invent `cmd` from the preamble and do not disable reasoning.
- Parallel-agent note: rebuild/redeploy the public tunnel/backend from current source, then recapture same-model Qwen35/Qwen3.6 direct/gateway/tunnel raw SSE with content deltas, reasoning events, function-call argument delta/done, final object consistency, valid output indices, and tool-result continuation.

# 2026-06-10 - Qwen35 public tunnel raw SSE recapture green

- Reduced blocker: #190/#192 same-model direct/gateway/tunnel Responses raw SSE parity for Qwen35/Qwen3.6 MXFP8-MTP.
- Public tunnel recapture: `build/responses-sse-captures-20260610/tunnel-qwen35-mxfp8-mtp-tool-recapture-after-strict-source-20260610.sse` from `https://testapi.adlabus.dev/v1/responses`.
- Parity proof: `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json`, `status=pass`, `missing_captures=[]`.
- Proven: direct, panel gateway, and public tunnel all report the same model `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`, preserve required `record_fact` args `{"value":"blue-cat"}`, include reasoning events with no reasoning-disable workaround, parse cleanly, have consistent final output, and have valid output item indices. Current direct/gateway use `message=[0]`, `reasoning=[1]`, `function_call=[2]`; fresh tunnel uses `message=[0]`, `function_call=[1]` with no conflict.
- Boundary: this clears the stale Qwen35 public tunnel duplicate-index blocker for this request/model. It does not prove all Qwen3-coder/N2/MiMo/Gemma parser families, tool-result continuation, installed-app parity, media, or full release readiness.

# 2026-06-10 - Qwen35 raw SSE checklist pointer refreshed

- Reduced blocker: stale full-objective checklist evidence for Qwen35 Responses raw SSE parity.
- Source edit: `tests/cross_matrix/run_full_release_objective_checklist.py` now points `QWEN35_RAW_SSE_PARITY` at `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json`.
- Regression guard: `tests/test_full_release_objective_checklist.py` asserts the current Qwen35 raw SSE artifact path so the checklist cannot silently fall back to the stale red 2026-06-09 tunnel capture.
- Checklist artifact: `build/current-full-release-objective-checklist-after-qwen35-public-sse-recapture-20260610.json`; parsed successfully with `status=open`, `release_ready=false`, and `failed_count=71`. No failed row references Qwen35 raw SSE after the pointer refresh.
- Verification: `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py -k 'qwen35_raw_sse or uses_current_qwen35_raw_sse_parity_contract'` passed (`2 passed`), Python JSON parse passed, and `git diff --check` passed.
- Boundary: this does not clear the generic Gemma4 E2B direct/gateway/tunnel raw SSE gate, which still points at `build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-e2b-after-parser-20260609.json` and remains red. Full release readiness remains open.
- Parallel-agent note: next Responses work should target the still-red generic/raw SSE family rows, especially tool-result continuation, kwargs passthrough, content/reasoning/function-call deltas, no raw XML leaks, final object consistency, and gateway/tunnel parity across Gemma/MiMo/N2/Qwen families.

# 2026-06-10 - Raw SSE parity now requires previous_response_id history guard

- Reduced blocker: Responses/API agentic-loop proof coverage for tool-result continuation.
- Source edit: `tests/cross_matrix/run_responses_raw_sse_parity_contract.py` now requires the no-heavy source contract check `responses_previous_response_history=true`; the raw SSE parity artifact exposes this as `responses_previous_response_history_guard`.
- Checklist edit: `tests/cross_matrix/run_full_release_objective_checklist.py` now fails raw SSE parity if the local previous-response history/tool-result continuation guard is missing or false.
- Proof updates:
  - `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json` remains `status=pass` and now records `responses_previous_response_history_guard=true`.
  - `build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-e2b-after-parser-20260609.json` remains `status=fail`, but local source guards are green, including `responses_previous_response_history_guard=true`.
  - `build/current-full-release-objective-checklist-after-qwen35-public-sse-recapture-20260610.json` remains `status=open`, `release_ready=false`, `failed_count=71`.
- Verification: `.venv/bin/python -m pytest -q tests/test_responses_raw_sse_parity_contract.py tests/test_full_release_objective_checklist.py -k 'raw_sse or qwen35_raw_sse or responses_raw_sse_parity'` passed (`24 passed`).
- Boundary: no runtime workaround, parser argument synthesis, reasoning-disable workaround, release step, PyPI publish, or N2 JANG_1L action was taken. The remaining Gemma4 E2B raw SSE blocker is public tunnel/model availability (`gemma4-e2b-sse` not advertised/served by tunnel), not local source history/tool-result coverage.
- Parallel-agent note: for the generic Gemma raw SSE row, redeploy/route the tunnel to the same `gemma4-e2b-sse` model or recapture against an actually advertised same Gemma model, then require content deltas, reasoning events, function-call args delta/done, final object consistency, valid output indices, and `responses_previous_response_history_guard=true`.

# 2026-06-10 - Qwen empty tool-call Chat/Responses harness fixed

- Reduced blocker: Qwen35/Qwen27 empty XML tool-call risk for OpenCode-style
  Chat Completions and Responses API harnesses.
- Source fix: `vmlx_engine/server.py::_parse_tool_calls_with_parser()` now
  keeps parser-init fallback behind the same request-schema filter and strips
  native tool markup residue when a parsed call is rejected for missing required
  args.
- Proof artifact:
  `build/current-noheavy-api-cache-contract-qwen-empty-tool-chat-responses-20260610.json`,
  `status=pass`, `missing_markers=[]`.
- Verification:
  focused server command passed with `7 passed`, and the full no-heavy
  API/cache contract passed. The `responses_streaming_tool_contracts` row now
  includes Responses streaming, Chat streaming, and shared nonstream parser
  coverage for the preamble plus empty XML `exec_command` shape.
- Proven: current source emits no executable `"arguments":"{}"` payload, no
  structured tool delta/item, and no raw invalid tool markup for the empty
  required-arg XML function shape. Valid XML parameter form still preserves
  `{"cmd":"ls /tmp"}`.
- Boundary: no live model load or tunnel redeploy was performed in this step.
  Keep Qwen35/Qwen27 live same-model harness recapture on the list for direct,
  gateway, and public tunnel if the backend changes. No release, PyPI, signing,
  notarization, media, MiMo, Gemma, or N2 JANG_1L action was taken.
- Commit/push: `658c9ab3 Harden Qwen empty tool-call filtering` was pushed to
  `origin/codex/pr-intake-manifest` and fast-forwarded to `origin/main`.

# 2026-06-10 - Gemma mixed-SWA storage quant runtime fix in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT cache/API proof. N2
  JANG_1L remains Eric-owned and is not being touched.
- Current constraint from Eric: do not use Python, local scripts, or wrappers
  to spawn subagents or delegate this lane's work to other agents. Direct local
  verification/artifact inspection is allowed when it is not subagent
  orchestration.
- Finding: CLI auto mode set stored prefix cache quantization to `q4`, but
  `MLLMScheduler` disabled it for every mixed-SWA VLM family before startup.
  That made Gemma4 12B JANG4M health report `storage_quantization.enabled=false`
  and kept `gemma4_12b_jang4m_nomedia_native_mixed_swa_cache` red.
- Source edit: keep auto q4/q8 storage quantization for Gemma4 mixed-SWA only;
  MiMo V2.5 and Step3.7 mixed-SWA remain explicit opt-in until their semantic
  parity rows are green.
- Source edit: health now reports storage quantization as
  `applies_to=full_attention_kv_only` with
  `metadata_policy=preserve_rotating_window_metadata`, because
  `RotatingKVCache` explicitly does not support quantization and must remain
  native.
- Verification: py_compile passed; focused MLLM rotating-cache preservation
  test passed; focused native-cache telemetry tests passed; focused release
  manifest mixed-SWA gate passed; focused full-release checklist tests passed.
- Live proof: loaded `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`
  from current source on port `8899` without an explicit
  `--kv-cache-quantization` flag. Startup kept auto `q4`, detected
  `40` RotatingKVCache layers and `8` KVCache layers, and initialized
  `kv_quant=q4`.
- Proof artifact:
  `build/current-gemma4-12b-jang4m-autoq4-mixed-swa-cache-live-20260610.json`,
  `status=pass`. It proves `mixed_swa_kv_v1`, `storage_quantization.enabled`
  true with `bits=4`, `applies_to=full_attention_kv_only`, rotating metadata
  preservation, generic TurboQuant KV inactive, second-turn
  `cached_tokens=20` with `cache_detail=paged+mixed_swa`, RAM cache tokens
  `20`, and block-disk L2 write/tokens `20`.
- Regenerated checklist:
  `build/current-full-release-objective-checklist-after-gemma12-autoq4-cache-20260610.json`
  remains `status=open`, but `failed_count` dropped to `67`; Gemma4 12B
  JANG4M no-media cache reuse/native mixed-SWA/block-L2 rows are green from the
  new cache proof.
- Still red: Gemma4 12B JANG4M no-media exact-code whitespace/status rows,
  Gemma QAT media/live rows, public Responses tunnel parity, MiMo exactness and
  media/L2 rows, installed-app parity, and release clearance. No release,
  signing, notarization, PyPI, or N2 JANG_1L action was taken.

# 2026-06-10 - Gemma QAT audio runtime gate in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT honest modality
  detection. N2 JANG_1L remains Eric-owned and is not being touched.
- Requested focus: do not advertise fake audio from config/token metadata;
  audio must be weight-backed and live-proven where claimed.
- Source edit: `tests/cross_matrix/run_gemma_qat_native_mxfp4_inventory_gate.py`
  now splits `audio_declared_by_config` from runtime `audio` support. Runtime
  audio is true only when `audio_tower.*` weights exist; metadata-only
  `embed_audio.*` no longer counts as audio support.
- Classification edit: when a required Gemma row declares audio but has no
  `audio_tower.*` weights, the inventory records
  `declared_required_modalities` separately and removes `audio` from the
  effective `required_modalities` instead of treating metadata-only audio as a
  native-audio proof.
- Pointer edits: current suite, objective checklist, objective digest, and
  release tracker now point at
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-audio-runtime-gate-20260610.json`.
- Regenerated artifacts:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-audio-runtime-gate-20260610.json`,
  `build/current-full-release-objective-checklist-after-gemma-audio-runtime-gate-20260610.json`,
  and `build/current-objective-proof-after-gemma-audio-runtime-gate-20260610.json`.
- Current proof state: inventory remains `status=open`, `missing_required_rows=[]`,
  `source_live_smoke_open_rows=[]`, and checklist remains `status=open` with
  `failed_count=67`. For 12B, audio is now recorded as
  `audio_declared_by_config=true`, `audio=false`,
  `audio_runtime_supported=false`; E2B/E4B keep audio runtime-supported because
  they have audio tower weights.
- Verification run so far: Gemma inventory tests `9 passed`; current-suite
  pointer tests `2 passed`; Gemma release-checklist focused tests `3 passed`;
  py_compile passed for the edited cross-matrix scripts.
- Not proven: live audio E2E for E2B/E4B, installed-app/UI/tunnel parity,
  release clearance, PyPI, signing, notarization, or any N2 JANG_1L claim.
- Other-agent action: keep E2B/E4B live audio rows active because they are
  weight-backed; do not claim 12B/26B/31B audio unless audio tower weights or a
  real runtime audio implementation is present and live-proven.

# 2026-06-10 - Gemma QAT audio runtime gate pushed

- Commit/push: `09b42d5b` (`Gate Gemma QAT audio by runtime weights`) was pushed to `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: Gemma QAT/native MXFP4 inventory/checklist honesty only. No release, signing, notarization, PyPI, public tunnel, MiMo, Qwen, or N2 JANG_1L action was included in this commit.
- Unrelated local state left alone: `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json` remains modified from other work and `node_modules/` remains untracked.

# 2026-06-10 - MiMo V2.5 vision head-dim parity fix in progress

- Directive check: allowed lane is MiMo V2.5 JANG/JANGTQ exactness/media
  proof. N2 JANG_1L remains Eric-owned and is not being touched.
- Finding: upstream MiMo vision runtime defaults missing `qk_channels` to `64`;
  vMLX defaulted to `hidden_size / num_heads`, which is `40` for the current
  1280-hidden/32-head MiMo bundle. Live qkv weight binding later inferred the
  correct value from real weights, but the source skeleton was not upstream
  faithful before that rescue path.
- Source edit: `vmlx_engine/models/mllm.py` now defaults MiMo vision q/k head
  dim to `64` when `qk_channels` is absent.
- Regression edit: `tests/test_mimo_v2_media_runtime.py` now asserts the
  upstream default `vision_head_dim == 64` and `blocks[0].attn.head_dim == 64`.
- Proof artifact:
  `build/current-mimo-v25-vision-head-dim-source-parity-20260610.json`.
- Verification: focused MiMo media runtime tests passed `7 passed`; py_compile
  passed; direct Torch-vs-MLX first-block parity probe used real
  `visual.blocks.0.*` weights and reported `vision_head_dim=64`,
  `block_head_dim=64`, `mean_abs_diff=0.000536009669303894`.
- Boundary: this is a real source-parity fix, but it does not claim the
  existing MiMo JANGTQ_2 visual semantic failures, literal exactness failures,
  installed-app rows, or release clearance are fixed. A fresh live media proof
  is still required after this patch.

# 2026-06-10 - MiMo V2.5 vision head-dim parity fix pushed

- Commit/push: `51477f05` (`Match MiMo vision qk head default`) was pushed to
  `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: MiMo source-parity fix plus proof artifact only. No release, signing,
  notarization, PyPI, public tunnel, Gemma, Qwen, or N2 JANG_1L action was
  included in this commit.
- Unrelated local state left alone:
  `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`
  remains modified from other work and `node_modules/` remains untracked.

# 2026-06-10 - MiMo JANGTQ2 live media/tools/cache proof after head-dim fix

- Directive check: allowed lane is MiMo V2.5 JANG/JANGTQ exactness/media/API/cache
  proof. N2 JANG_1L remains Eric-owned and was not touched.
- Live server: current source served
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` on port `8877`
  with `--mllm`, `--max-tokens 64`, greedy defaults, and
  `VMLINUX_L2_CACHE_DIR=build/mimo-jangtq2-media-live-l2-after-head-dim-20260610`.
  The server was stopped after the proof; no `8877` listener remains.
- Proof artifact:
  `build/current-mimo-v25-jangtq2-live-media-tools-cache-after-head-dim-20260610.json`.
  Raw responses are under
  `build/mimo-v25-after-head-dim-live-requests-20260610/`.
- Proven:
  - current source loads the MiMo JANGTQ_2 bundle after the head-dim fix;
  - preserved visual/audio/speech weights bind into runtime;
  - app icon image returns visible text `vMLX`;
  - video and audio payloads reach the media runtime path;
  - Chat required-tool emits valid OpenAI tool-call structure;
  - Responses required-tool stream emits function-call argument delta/done
    events and valid output indices (`message=0`, `function_call=1`);
  - same-process paged native MiMo mixed-SWA cache reuse is active
    (`cache_hit_tokens=280`, last Responses request `cached_tokens=253`);
  - generic TurboQuant KV is not substituted for MiMo asymmetric SWA.
- Still red:
  - literal exactness: `MIMO-OK` became `MIMOOK` and `blue-cat` became `blue cat`;
  - red video semantic answer returned `White.`;
  - audio exactness/hygiene is not green: default output leaked planning prose
    and explicit no-thinking output denied receiving audio despite routed
    input_audio logs;
  - block-disk L2 write/fresh-process restore were not enabled or proven in
    this launch;
  - auto-tool, no-tool, tool-result continuation, cancellation cleanup,
    largest-context tail, UI, installed-app, package, signing, notarization,
    and release clearance remain open.
- Other-agent action: do not mark MiMo exactness or media green from this proof.
  Next useful MiMo work is artifact/logit/quant-contract or runtime decode
  diagnosis, plus a separate explicit L2 restart proof if needed. Do not hide
  failures through parser semantic repair, tool-argument rewriting, prompt-only
  folding, hidden sampling overrides, or fake release wording.
- Carry-forward instruction written into `AGENTS.md`: Qwen3.6/Qwen-coder
  empty-args and Responses deltas remain active across 27B/35B style XML
  tool-call dialects; prove same-model direct/gateway/tunnel raw SSE with
  reasoning on and do not synthesize missing tool parameters.

# 2026-06-10 - MiMo JANGTQ2 live media boundary pushed

- Commit/push: `baa311e5` (`Record MiMo JANGTQ2 live media boundary`) was
  pushed to `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: proof/status/tracker/AGENTS only. No runtime code, release, signing,
  notarization, PyPI, public download, Gemma, Qwen live tunnel, or N2 JANG_1L
  action was included.
- Unrelated local state left alone:
  `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`
  remains modified from other work and `node_modules/` remains untracked.

# 2026-06-10 - Qwen Responses raw-SSE release gate refresh in progress

- Directive check: allowed lane is Qwen/Qwen3.6 Responses/tool/reasoning
  streaming parity and cross-family parser/API contract. N2 JANG_1L remains
  off-limits and no release/sign/notarize/PyPI action is being taken.
- Finding: the Qwen empty-args source boundary and Qwen35 same-model
  direct/gateway/tunnel raw SSE proof are already present and green in current
  source, but the generic full-release Responses raw-SSE gate still consumed
  the older Gemma4 E2B tunnel-unavailable artifact.
- Source edit: `tests/cross_matrix/run_full_release_objective_checklist.py`
  now points `RESPONSES_RAW_SSE_PARITY` and `DEFAULT_OUT` at the current
  Qwen35/Qwen empty-args refresh lane. `tests/test_full_release_objective_checklist.py`
  was updated to pin that pointer.
- Tracker edit: the release tracker now records the Qwen35 same-model raw-SSE
  pass as the current Responses raw-SSE release-gate evidence and keeps Gemma4
  E2B tunnel availability separate.
- Regenerated artifact:
  `build/current-full-release-objective-checklist-after-qwen-mimo-gemma-refresh-20260610.json`
  is `status=open`, `failed_count=59`.
- Verification:
  - `.venv/bin/python tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-qwen-mimo-gemma-refresh-20260610.json`
    completed with expected nonzero exit because release remains open and
    printed `failed_count=59`.
  - `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py -k 'responses_raw_sse_parity or qwen35_raw_sse_parity'`
    passed `2 passed`.
  - `python3 -m py_compile tests/cross_matrix/run_full_release_objective_checklist.py`
    passed.
- Current boundary: Qwen empty-args/direct-gateway-tunnel raw-SSE issue is
  release-gate green for the current Qwen35 proof. Release remains blocked by
  real open rows including MiMo exactness/media/L2, Gemma QAT full media/UI,
  N2 JANG_1L (Eric-owned/off-limits here), Step/LFM/Nemotron/DSV4, package,
  signing, notarization, and public release gates.

# 2026-06-10 - Qwen Responses SSE gate refresh pushed

- Commit/push: `f199c893` (`Consume current Qwen Responses SSE proof`) was
  pushed to `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: release-gate proof pointer, refreshed checklist artifact, tracker,
  and status notes only. No model server launch, release, signing,
  notarization, PyPI, public download, or N2 JANG_1L action was included.
- Current checklist artifact:
  `build/current-full-release-objective-checklist-after-qwen-mimo-gemma-refresh-20260610.json`
  remains `status=open`, `failed_count=59`.

# 2026-06-10 - Gemma 12B JANG_4M exact-code prompt fix in progress

- Directive check: allowed lane is Gemma JANG/MXFP parser/runtime/cache/API
  proof. N2 JANG_1L remains off-limits and no release/sign/notarize/PyPI
  action is being taken.
- Finding: the checklist Gemma4 12B JANG_4M no-media row was red from
  `exact_code_whitespace`: output was
  `def add(a, b):\n    return a + b\n print(add(2, 3))`.
- Live isolation: served `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`
  on port `8890` with prefix/paged/block cache disabled and greedy defaults.
  The old prompt reproduced the extra leading space; a clearer prompt requiring
  the third line to start at column 1 returned exact code. This excludes cache
  reuse as the cause and identifies an ambiguous prompt contract, not parser
  rewriting.
- Source edit: `bench/all_local_model_smoke.py` exact-code prompt now says the
  third line must start at column 1 with no leading space. Validation remains
  strict and no output repair was added.
- Live proof: focused Gemma 12B JANG_4M no-media smoke with tools and L2 restart
  wrote
  `build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-code-column-prompt-20260610/`.
  Both matching rows passed with `failures=0`; exact-code output is exactly
  `def add(a, b):\n    return a + b\nprint(add(2, 3))`.
- Source/proof pointer edit: full release checklist now consumes the fresh
  passing Gemma 12B JANG_4M artifact.
- Regenerated artifact:
  `build/current-full-release-objective-checklist-after-gemma12-code-column-prompt-20260610.json`
  is `status=open`, `failed_count=57`.
- Verification:
  - `.venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --out build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-code-column-prompt-20260610 --port 8890 --only gemma-4-12B-it-JANG_4M --no-media --include-tools --include-l2-restart --load-timeout-s 240 --request-timeout-s 240`
    exited `0`; both matching Gemma 12B JANG_4M rows passed.
  - `.venv/bin/python tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-gemma12-code-column-prompt-20260610.json`
    completed with expected nonzero exit because release remains open and
    printed `failed_count=57`.
  - `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py -k 'gemma4_12b_jang4m_nomedia or responses_raw_sse_parity or qwen35_raw_sse_parity'`
    passed `3 passed`.
  - `python3 -m py_compile bench/all_local_model_smoke.py tests/cross_matrix/run_full_release_objective_checklist.py`
    passed.
- Boundary: Gemma 12B JANG_4M no-media exact-code row is green from current
  source. Gemma QAT/native MXFP4 full live media/UI/installed-app rows remain
  open; this is not release clearance.

# 2026-06-10 - Gemma exact-code prompt fix pushed

- Commit/push: `2ddbba65` (`Clarify Gemma exact code smoke prompt`) was pushed
  to `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: Gemma no-media exact-code prompt contract, focused Gemma proof
  artifacts, release-checklist pointer/artifact, tracker, and status notes.
  No release, signing, notarization, PyPI, public download, or N2 JANG_1L
  action was included.
- Current checklist:
  `build/current-full-release-objective-checklist-after-gemma12-code-column-prompt-20260610.json`
  is `status=open`, `failed_count=57`.

# 2026-06-10 - AGENTS.md current control contract update in progress

- User request: write the current operating constraints into `AGENTS.md` so
  future continuations do not ignore the active goal, N2 JANG_1L stop
  condition, parser/API priority, or no-subagent rule.
- Directive check: documentation/control-plane update only. Allowed lanes are
  still MiMo, Gemma, Qwen parser/API/gateway, and N2 non-JANG_1L. N2 JANG_1L
  remains Eric-owned/off-limits. No release/sign/notarize/PyPI/download action
  is being taken.
- Edit: `AGENTS.md` mandatory continuation loop now explicitly requires
  reading `.agents/CODEX_ACTIVE_DIRECTIVES_20260610.md`, `.agents/STATUS.md`,
  and the latest checklist/proof artifact before action; writing every
  movement with proven/not-proven/blocker/no-claim/other-agent details; keeping
  N2 JANG_1L off-limits unless Eric reopens it; and not entering release steps
  unless the current-turn release lock is explicitly lifted.
- Boundary: this is not a model/runtime proof and does not clear any release
  checklist row. It is a control-doc fix to prevent drift before the next live
  MiMo/Gemma/Qwen proof or fix.

# 2026-06-10 - AGENTS.md control contract pushed

- Commit/push: `c630b1d9` (`Record active agent control contract`) was pushed
  to `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: `AGENTS.md`, `.agents/STATUS.md`, and `.agents/LOG.md` only.
- Boundary: no release/sign/notarize/PyPI/download action, no model server
  launch, and no N2 JANG_1L action. Next work should resume from the latest
  checklist and reduce a live MiMo/Gemma/Qwen blocker.

# 2026-06-10 - MiMo JANGTQ2 current media proof accounting in progress

- Directive check: allowed lane is MiMo V2.5 JANGTQ_2 media/cache/API/UI proof
  accounting from current artifacts. N2 JANG_1L remains off-limits. No release,
  signing, notarization, PyPI, public download, or model launch is being done.
- Source edit: `tests/cross_matrix/run_full_release_objective_checklist.py`
  now consumes current MiMo JANGTQ2 media/runtime proof artifacts:
  `build/current-mimo-v25-jangtq2-media-runtime-source-proof-20260610.json`,
  `build/current-mimo-v25-jangtq2-video-audio-source-proof-20260610.json`,
  `build/current-real-ui-dev-app-mimo-v25-jangtq2-responses-tools-cache-20260610.json`,
  and the newer no-source exactness classifier
  `build/current-mimo-v2-no-source-exactness-classifier-after-devapp-jangtq2-exactness-20260610.json`.
- Proven by current checklist:
  `mimo_media_runtime_implementation=true`, `mimo_mimo_media_wired=true`,
  `mimo_jangtq2_current_source_media_runtime=true`,
  `mimo_jangtq2_current_source_video_audio_routes=true`, and
  `mimo_jangtq2_dev_app_responses_tools_cache=true`.
- Still red by current checklist: `mimo_local_release_clearance`,
  `mimo_decode_speed_target`, `mimo_artifact_exactness`, and
  `mimo_jangtq2_media_semantics_release_quality`.
- Regenerated artifact:
  `build/current-full-release-objective-checklist-after-mimo-current-media-accounting-20260610.json`
  is `status=open`, `failed_count=56`.
- Boundary: this clears stale "media implementation missing/unwired" accounting
  only. It does not claim MiMo exactness, visual semantic quality, audio
  semantic quality, fresh-process L2 restore, installed-app parity, or release
  readiness. Next other-agent action: focus MiMo on artifact/logit/quant
  contract or runtime decode diagnosis and semantic media quality, not parser
  repair, cache chasing, or release wording.

# 2026-06-10 - MiMo JANGTQ2 current media proof accounting pushed

- Commit/push: `53771351` (`Consume current MiMo media proof`) was pushed to
  `origin/codex/pr-intake-manifest` and `origin/main`.
- Scope: release-checklist proof consumption, focused checklist test fixture,
  regenerated checklist artifact, and status/log notes.
- Current checklist:
  `build/current-full-release-objective-checklist-after-mimo-current-media-accounting-20260610.json`
  remains `status=open`, `failed_count=56`.
- No release/sign/notarize/PyPI/download action, no model server launch, and no
  N2 JANG_1L action was included.

# 2026-06-10 - Gemma E2B QAT JANG4M full-media source proof in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT VL/video/cache/API/UI
  proof and honest modality gating. N2 JANG_1L remains off-limits. No release,
  signing, notarization, PyPI, public download, or package action is being
  done.
- Live proof command:
  `.venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --out build/current-all-local-model-smoke-gemma4-e2b-qat-jang4m-fullmedia-tools-l2-20260610 --port 8890 --only gemma-4-E2B-it-qat-JANG_4M --include-tools --include-l2-restart --load-timeout-s 240 --request-timeout-s 240`
- Live proof result: real `/Users/eric/models/JANGQ-AI/gemma-4-E2B-it-qat-JANG_4M`
  loaded and passed `status=pass`, `failures=0`.
- Proven: visible text, cache first miss/second hit, multi-turn recall,
  reasoning separation, required tool call with exact `{"value": "blue-cat"}`,
  tool-result continuation, exact JSON, exact code whitespace, image blue/red,
  video blue through vision, audio blue, post-media text recovery, Gemma4
  parser/reasoning parser, JANG affine Metal NA dispatch, native mixed-SWA
  cache with generic TurboQuant KV disabled, block L2 writes, and fresh-process
  L2 restore with `cache_detail=paged+mixed_swa+disk`.
- Source edit: Gemma QAT/native inventory now records
  `source_fullmedia_smoke` for `gemma4_e2b_qat_jang4m` and validates direct
  result artifacts with top-level `requests`.
- Regenerated artifacts:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-e2b-jang4m-fullmedia-20260610.json`
  and
  `build/current-full-release-objective-checklist-after-gemma-e2b-jang4m-fullmedia-20260610.json`.
- Boundary: E2B QAT JANG4M source full-media proof is green, but the Gemma
  release row remains open because Responses streaming/non-streaming, UI,
  installed-app parity, and the remaining Gemma QAT/JANG4M rows are not fully
  cleared by this single source smoke. Do not claim Gemma release clearance.

# 2026-06-10 - AGENTS.md no-subagent rule tightened

- User request: record the no-Python-subagent/no-delegation constraint in
  `AGENTS.md` so future continuations do not use recursive agent behavior while
  working the model/runtime/release lane.
- Directive check: documentation/control-plane update only. N2 JANG_1L remains
  Eric-owned/off-limits. No release, signing, notarization, PyPI, public
  download, package, or model launch action was taken.
- Edit: added a dedicated `No subagent delegation` section to `AGENTS.md`.
  It forbids Python, shell wrappers, MCP tools, orchestration scripts, or
  hidden helper processes from spawning, managing, prompting, or monitoring
  subagents for this lane's work.
- Allowed boundary recorded: ordinary Python or shell commands remain allowed
  for direct local verification, artifact inspection, test execution, proof
  generation, and source maintenance when they do not create or manage
  subagents.
- Proven: the active worktree control doc now contains the explicit
  no-subagent rule.
- Not proven: no runtime/model/API/UI/cache row changed or was exercised by
  this documentation edit.

# 2026-06-10 - Gemma E4B QAT JANG4M full-media source proof in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT VL/video/cache/API/UI
  proof and honest modality gating. N2 JANG_1L remains off-limits. No release,
  signing, notarization, PyPI, public download, or package action is being
  taken.
- Blocker being reduced: Gemma QAT/native MXFP4 release checklist still lacks
  current full-media source proof for QAT JANG4M rows beyond E2B.
- Planned direct proof command:
  `.venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --out build/current-all-local-model-smoke-gemma4-e4b-qat-jang4m-fullmedia-tools-l2-20260610 --port 8890 --only gemma-4-E4B-it-qat-JANG_4M --include-tools --include-l2-restart --load-timeout-s 240 --request-timeout-s 240`
- Boundary: this is not release/signing/notarization/PyPI/download work and
  must not be claimed as full Gemma release clearance even if it passes.
- Live proof result: real `/Users/eric/models/JANGQ-AI/gemma-4-E4B-it-qat-JANG_4M`
  loaded and passed with `status=pass`, `failures=0`.
- Proof artifacts:
  `build/current-all-local-model-smoke-gemma4-e4b-qat-jang4m-fullmedia-tools-l2-20260610/summary.json`
  and
  `build/current-all-local-model-smoke-gemma4-e4b-qat-jang4m-fullmedia-tools-l2-20260610/JANGQ_gemma-4-E4B-it-qat-JANG_4M/result.json`.
- Proven: visible output, cache first miss/second hit with
  `cache_detail=paged+mixed_swa`, multi-turn recall, reasoning separation,
  required tool call with exact `{"value": "blue-cat"}`, tool-result
  continuation, exact JSON, exact code whitespace, image blue/red, video blue
  through vision, audio blue, post-media text recovery, Gemma4 parser/reasoning
  parser, JANG affine Metal NA dispatch, native mixed-SWA cache with generic
  TurboQuant KV disabled, block L2 write, and fresh-process L2 restore with
  `cache_detail=paged+mixed_swa+disk`.
- Source/proof pointer edit: Gemma QAT/native inventory now records
  `source_fullmedia_smoke.status=pass` for `gemma4_e4b_qat_jang4m`; full
  release checklist and current regression suite consume
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-e4b-jang4m-fullmedia-20260610.json`.
- Regenerated artifacts:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-e4b-jang4m-fullmedia-20260610.json`
  and
  `build/current-full-release-objective-checklist-after-gemma-e4b-jang4m-fullmedia-20260610.json`.
- Verification: `python3 -m py_compile` passed for touched gate files;
  `.venv/bin/python -m pytest -q tests/test_gemma_qat_native_mxfp4_inventory_gate.py tests/test_full_release_objective_checklist.py -k 'gemma_qat or gemma4_e4b or full_release_objective_checklist'`
  passed `28 passed`; `git diff --check` passed. Full checklist remains
  expected-open with `failed_count=56`.
- Not proven: E4B Responses streaming/non-streaming, UI, installed-app parity,
  and full Gemma release clearance remain open. QAT JANG4M rows for 12B, 26B,
  and 31B still lack current full-media source proof.

# 2026-06-10 - Gemma 12B QAT JANG4M full-media honest-audio proof in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT VL/video/cache/API/UI
  proof and honest modality gating. N2 JANG_1L remains off-limits. No release,
  signing, notarization, PyPI, public download, or package action is being
  taken.
- Blocker being reduced: Gemma 12B QAT JANG4M has no current full-media source
  proof. Its local artifact advertises audio metadata but has no audio tower,
  so audio must be honestly gated, not faked as weight-backed.
- Source gate edit: `_source_fullmedia_smoke_status` now receives the
  row-specific `required_modalities`; audio labels are required only when audio
  remains required for that artifact. E2B/E4B still require audio; 12B QAT
  JANG4M does not because `audio_weight_backed=false`.
- Planned direct proof command:
  `.venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --out build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-fullmedia-tools-l2-20260610 --port 8890 --only gemma-4-12B-it-qat-JANG_4M --include-tools --include-l2-restart --load-timeout-s 240 --request-timeout-s 240`
- Boundary: this must prove vision/video/tools/cache/L2 plus honest no-audio
  handling. It must not claim audio runtime support without audio weights.
- Live proof result: real `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-qat-JANG_4M`
  loaded and passed with `status=pass`, `failures=0`.
- Proof artifacts:
  `build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-fullmedia-tools-l2-20260610/summary.json`
  and
  `build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-fullmedia-tools-l2-20260610/JANGQ_gemma-4-12B-it-qat-JANG_4M/result.json`.
- Proven: visible output, cache first miss/second hit with
  `cache_detail=paged+mixed_swa`, multi-turn recall, reasoning separation,
  exact required tool args `{"value": "blue-cat"}`, tool-result continuation,
  exact JSON, exact code whitespace, blue/red image, blue video through vision,
  post-media text recovery, Gemma4 parser/reasoning parser, native mixed-SWA
  cache with generic TurboQuant KV disabled, block L2 write, and fresh-process
  L2 restore with `cache_detail=paged+mixed_swa+disk`.
- Honest audio boundary: the 12B QAT JANG4M artifact has audio metadata but no
  audio tower weights. The proof did not include `audio_blue`; the regenerated
  inventory records `source_fullmedia_smoke.requires_audio=false` and
  `request_count=14`. This is honest no-audio gating, not audio runtime proof.
- Source/proof pointer edit: Gemma QAT/native inventory now records
  `source_fullmedia_smoke.status=pass` for `gemma4_12b_qat_jang4m`; full
  release checklist and current regression suite consume
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-12b-jang4m-fullmedia-20260610.json`.
- Regenerated artifacts:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-12b-jang4m-fullmedia-20260610.json`
  and
  `build/current-full-release-objective-checklist-after-gemma-12b-jang4m-fullmedia-20260610.json`.
- Verification: `python3 -m py_compile` passed for touched gate files;
  `.venv/bin/python -m pytest -q tests/test_gemma_qat_native_mxfp4_inventory_gate.py tests/test_full_release_objective_checklist.py -k 'gemma_qat or gemma4_12b or gemma4_e4b or full_release_objective_checklist'`
  passed `28 passed`; `git diff --check` passed. Full checklist remains
  expected-open with `failed_count=56`.
- Not proven: 12B Responses streaming/non-streaming, UI, installed-app parity,
  audio runtime support, and full Gemma release clearance remain open. QAT
  JANG4M rows for 26B and 31B still lack current full-media source proof.

# 2026-06-10 - Gemma 26B QAT JANG4M full-media honest-audio proof in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT VL/video/cache/API/UI
  proof and honest modality gating. N2 JANG_1L remains off-limits. No release,
  signing, notarization, PyPI, public download, or package action is being
  taken.
- Blocker being reduced: Gemma 26B QAT JANG4M has no current full-media source
  proof. Its local artifact has no audio tower, so audio must stay honestly
  absent while text/vision/video/cache/tools/L2 are proven.
- Planned direct proof command:
  `.venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --out build/current-all-local-model-smoke-gemma4-26b-qat-jang4m-fullmedia-tools-l2-20260610 --port 8890 --only gemma-4-26B-A4B-it-qat-JANG_4M --include-tools --include-l2-restart --load-timeout-s 300 --request-timeout-s 300`
- Boundary: this must not claim audio runtime support or full Gemma release
  clearance.
- Live proof result: real `/Users/eric/models/JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M`
  loaded and passed with `status=pass`, `failures=0`.
- Proof artifacts:
  `build/current-all-local-model-smoke-gemma4-26b-qat-jang4m-fullmedia-tools-l2-20260610/summary.json`
  and
  `build/current-all-local-model-smoke-gemma4-26b-qat-jang4m-fullmedia-tools-l2-20260610/JANGQ_gemma-4-26B-A4B-it-qat-JANG_4M/result.json`.
- Proven: visible output, cache first miss/second hit with
  `cache_detail=paged+mixed_swa`, multi-turn recall, reasoning separation,
  exact required tool args `{"value": "blue-cat"}`, tool-result continuation,
  exact JSON, exact code whitespace, blue/red image, blue video through vision,
  post-media text recovery, Gemma4 parser/reasoning parser, native mixed-SWA
  cache with generic TurboQuant KV disabled, block L2 write, and fresh-process
  L2 restore with `cache_detail=paged+mixed_swa+disk`.
- Honest audio boundary: 26B has no audio tower. The proof did not include
  `audio_blue`; the regenerated inventory records
  `source_fullmedia_smoke.requires_audio=false` and `request_count=14`.
- Source/proof pointer edit: Gemma QAT/native inventory now records
  `source_fullmedia_smoke.status=pass` for `gemma4_26b_qat_jang4m`; full
  release checklist and current regression suite consume
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-26b-jang4m-fullmedia-20260610.json`.
- Regenerated artifacts:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-26b-jang4m-fullmedia-20260610.json`
  and
  `build/current-full-release-objective-checklist-after-gemma-26b-jang4m-fullmedia-20260610.json`.
- Verification: `python3 -m py_compile` passed for touched gate files;
  `.venv/bin/python -m pytest -q tests/test_gemma_qat_native_mxfp4_inventory_gate.py tests/test_full_release_objective_checklist.py -k 'gemma_qat or gemma4_26b or gemma4_12b or full_release_objective_checklist'`
  passed `28 passed`; `git diff --check` passed. Full checklist remains
  expected-open with `failed_count=56`.
- Not proven: 26B Responses streaming/non-streaming, UI, installed-app parity,
  audio runtime support, and full Gemma release clearance remain open. QAT
  JANG4M 31B still lacks current full-media source proof.

# 2026-06-10 - Gemma 31B QAT JANG4M full-media honest-audio proof in progress

- Directive check: allowed lane is Gemma JANG/MXFP/QAT VL/video/cache/API/UI
  proof and honest modality gating. N2 JANG_1L remains off-limits. No release,
  signing, notarization, PyPI, public download, or package action is being
  taken.
- Blocker being reduced: Gemma 31B QAT JANG4M is the remaining QAT JANG4M row
  without current full-media source proof. Its local artifact has no audio
  tower, so audio must stay honestly absent while text/vision/video/cache/tools
  and L2 are proven.
- Planned direct proof command:
  `.venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --out build/current-all-local-model-smoke-gemma4-31b-qat-jang4m-fullmedia-tools-l2-20260610 --port 8890 --only gemma-4-31B-it-qat-JANG_4M --include-tools --include-l2-restart --load-timeout-s 300 --request-timeout-s 300`
- Boundary: this must not claim audio runtime support, Responses/UI parity, or
  full Gemma release clearance.
- Live proof result: real `/Users/eric/models/JANGQ-AI/gemma-4-31B-it-qat-JANG_4M`
  loaded and passed with `status=pass`, `failures=0`.
- Proof artifacts:
  `build/current-all-local-model-smoke-gemma4-31b-qat-jang4m-fullmedia-tools-l2-20260610/summary.json`
  and
  `build/current-all-local-model-smoke-gemma4-31b-qat-jang4m-fullmedia-tools-l2-20260610/JANGQ_gemma-4-31B-it-qat-JANG_4M/result.json`.
- Proven: visible output, cache first miss/second hit with
  `cache_detail=paged+mixed_swa`, multi-turn recall, reasoning separation,
  exact required tool args `{"value": "blue-cat"}`, tool-result continuation,
  exact JSON, exact code whitespace, blue/red image, blue video through vision,
  post-media text recovery, Gemma4 parser/reasoning parser, native mixed-SWA
  cache with generic TurboQuant KV disabled, block L2 write, and fresh-process
  L2 restore with `cache_detail=paged+mixed_swa+disk`.
- Honest audio boundary: 31B has no audio tower. The proof did not include
  `audio_blue`; the regenerated inventory records
  `source_fullmedia_smoke.requires_audio=false` and `request_count=14`.
- Source/proof pointer edit: Gemma QAT/native inventory now records
  `source_fullmedia_smoke.status=pass` for all five QAT JANG4M rows:
  E2B, E4B, 12B, 26B, and 31B. Full release checklist and current regression
  suite consume
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-all-jang4m-fullmedia-20260610.json`.
- Regenerated artifacts:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-all-jang4m-fullmedia-20260610.json`
  and
  `build/current-full-release-objective-checklist-after-gemma-all-jang4m-fullmedia-20260610.json`.
- Verification: `python3 -m py_compile` passed for touched gate files;
  `.venv/bin/python -m pytest -q tests/test_gemma_qat_native_mxfp4_inventory_gate.py tests/test_full_release_objective_checklist.py -k 'gemma_qat or gemma4_31b or gemma4_26b or full_release_objective_checklist'`
  passed `28 passed`; `git diff --check` passed. Full checklist remains
  expected-open with `failed_count=56`.
- Not proven: QAT JANG4M Responses streaming/non-streaming direct/gateway/tunnel
  parity, UI, installed-app parity, full Gemma release clearance, and QAT/native
  MXFP4 rows outside these QAT JANG4M source smokes remain open.

# 2026-06-10 - Qwen35 raw SSE parity recheck

- Directive check: allowed lane is Qwen/Qwen3.6/Qwen-coder
  Responses/tool/reasoning streaming parity. N2 JANG_1L remains off-limits. No
  release, signing, notarization, PyPI, public download, package, or model
  launch action is being taken.
- Blocker being reduced: Qwen XML-tool empty-args / output-index /
  reasoning-delta streaming correctness for opencode/Codex-style harnesses.
- Current evidence read:
  `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json`.
- Proven by current artifact: direct local server, panel gateway, and tunnel
  are same-model for `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`; all three have
  authoritative function-call args `{"value": "blue-cat"}`, matching
  `record_fact`, argument delta/done events, final response consistency,
  required reasoning events, and no reasoning-disable workaround. Local source
  guards report `local_empty_xml_arguments_fail_closed=true`,
  `local_output_index_ordering_guard=true`, and gateway request kwargs preserve
  `enable_thinking=true`, `tool_choice=required`, max output, top-p/top-k, and
  tool count.
- Boundary found during raw SSE inspection: direct and gateway emit a separate
  reasoning output item at `output_index=1` and the function call at
  `output_index=2`. The public tunnel capture still streams reasoning summary
  deltas on the message item at `output_index=0` and does not emit a separate
  reasoning output item before the function call at `output_index=1`, although
  the final response includes a reasoning item and the function args are
  correct. This is a tunnel/public parity boundary, not a direct/gateway source
  parser failure from the current artifact.
- Not claimed: full direct/gateway/tunnel reasoning-item shape parity is not
  claimed from this artifact. Qwen35 direct/gateway source behavior is clean
  for the reported empty-args failure; public tunnel reasoning item indexing
  should be recaptured after the public route is confirmed to run the same
  source.

# 2026-06-10 - AGENTS guard updated for parser/API and no-subagent constraints

- Directive check: allowed lane is documentation/status guard maintenance for
  the active parser/API/runtime proof work. N2 JANG_1L remains off-limits. No
  model launch, release, signing, notarization, PyPI, public download, package,
  or source runtime fix is being taken in this movement.
- Request: write the active instructions into `AGENTS.md`, including the
  no-subagent constraint and the Qwen/Qwen-coder empty tool-arguments issue,
  and force this lane to remember all parser/API/gateway/tool/reasoning
  streaming work.
- Action: updated active `AGENTS.md` in
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`, not deprecated
  `/Users/eric/vmlx`.
- Written into guard: the Qwen3.6/Qwen-coder 27B/35B-style XML empty-arguments
  report is an active fix/proof item, but the proposed root cause must not be
  trusted without same-model raw output. Missing required tool args must fail
  closed; do not synthesize `cmd`, infer args from visible preamble, disable
  reasoning, silently drop tool calls, or patch semantic values after parser
  failure.
- Written into guard: cross-family parser/API proof must cover auto tool,
  required tool, no-tool, tool-result continuation, content deltas, reasoning
  deltas, function-call args delta/done, final response object consistency,
  request kwargs passthrough, parser selection, cache reuse telemetry, gateway
  and tunnel route parity, and raw leak checks across Qwen, Qwen-coder, Gemma4,
  MiMo/think-XML, MiniMax, DeepSeek/R1-style think parsers, XML function-call
  parsers, and other family-specific parser paths.
- Not proven: this is guard/status documentation only. It does not itself prove
  opencode/Codex harness usability, public tunnel parity, MiMo exactness,
  Gemma UI/installed-app parity, N2 JANGTQ media, or release readiness.
- Other-agent action: use `AGENTS.md` as the standing checklist before touching
  parser/API/gateway work; keep public tunnel reasoning-item shape red until
  recaptured from current source; keep MiMo exactness out of parser/JSON repair;
  keep N2 JANG_1L out of this lane unless Eric explicitly reopens it.

# 2026-06-10 - XML/Qwen required tool-argument fail-closed runtime fix

- Directive check: allowed lane is Qwen/Qwen3.6/Qwen-coder and cross-family
  parser/API/tool-reasoning correctness. N2 JANG_1L remains off-limits. No
  model launch, release, signing, notarization, PyPI, public download, or
  package action is being taken.
- Blocker being reduced: opencode/Codex-style harnesses must not receive
  `arguments: {}` when a model emits a tool marker/function name but omits
  required parameters, including the reported text-preamble plus
  `<tool_call><function=exec_command></function></tool_call>` shape.
- Root cause found in source: XML-dialect parsers could emit structured tool
  calls with empty `{}` arguments when a `<function=...>` block had no
  `<parameter=...>` tags or JSON args. Affected direct parser surfaces included
  `xml_function`, `nemotron`, and `step3p5`; Qwen JSON-in-XML could also accept
  `{"arguments": {}}` before schema-required validation.
- Fix: added shared parser helper
  `_arguments_satisfy_required_schema()` and wired it into Qwen,
  XMLFunction/MiMo, Nemotron, and Step3p5 parsers. If the request schema has
  required parameters and the parsed arguments omit them or provide empty
  strings, the parser now rejects the tool call instead of emitting `{}`.
- Server boundary proof: added regression showing
  `_parse_tool_calls_with_parser()` with `tool_call_parser="xml_function"` and
  required `exec_command.cmd` returns `tool_calls is None` for the reported
  empty-function output, not a `ToolCall(arguments="{}")`.
- Verification:
  `python3 -m py_compile vmlx_engine/tool_parsers/abstract_tool_parser.py vmlx_engine/tool_parsers/qwen_tool_parser.py vmlx_engine/tool_parsers/xml_function_tool_parser.py vmlx_engine/tool_parsers/nemotron_tool_parser.py vmlx_engine/tool_parsers/step3p5_tool_parser.py`
  passed.
- Verification:
  `.venv/bin/python -m pytest -q tests/test_xml_function_tool_parser.py tests/test_step3p5_tool_parser.py tests/test_tool_parsers.py tests/test_engine_audit.py -k 'empty_arguments or empty_function or required_schema or qwen_issue_192 or xml_function_empty_required_args_fail_closed_at_server_boundary or QwenToolParser or NemotronToolParser or Step3p5ToolParser or XMLFunctionToolParser'`
  passed `42 passed`.
- Verification:
  `.venv/bin/python -m pytest -q tests/test_xml_function_tool_parser.py tests/test_step3p5_tool_parser.py tests/test_tool_parsers.py`
  passed `119 passed`; `git diff --check` passed.
- Proven: parser and server boundary fail closed for schema-required missing
  arguments across the affected XML/Qwen parser paths. Valid no-argument tools
  remain allowed when the request schema has no required parameters.
- Not proven: this is not a fresh same-model direct/gateway/tunnel raw SSE
  model recapture, not public tunnel parity, not live opencode end-to-end, not
  MiMo exactness/media proof, not Gemma UI/installed-app proof, and not release
  readiness.
- Other-agent action: recapture same-model Qwen/Qwen-coder direct, gateway, and
  tunnel raw SSE with reasoning enabled after this commit lands; expected
  behavior for truly missing required args is fail-closed/no tool call rather
  than `arguments: {}`. Keep content/reasoning deltas, args delta/done,
  output-index ordering, tool-result continuation, and cache telemetry in the
  recapture.

# 2026-06-10 - Qwen35 direct/gateway raw SSE recaptured after fail-closed fix

- Directive check: allowed lane is Qwen/Qwen3.6/Qwen-coder
  Responses/tool/reasoning streaming parity. N2 JANG_1L remains off-limits. No
  release, signing, notarization, PyPI, public download, or package action was
  taken.
- Blocker being reduced: prove current source direct server and panel gateway
  still preserve tool args/reasoning/output-index/cache behavior after the
  missing-required-args fail-closed parser fix.
- Command:
  `.venv/bin/python tests/cross_matrix/run_qwen35_responses_raw_sse_capture.py --out build/current-responses-raw-sse-parity-qwen35-direct-gateway-after-missing-required-args-failclosed-20260610.json --direct-sse build/responses-sse-captures-20260610/direct-qwen35-mxfp8-mtp-tool-after-missing-required-args-failclosed-20260610.sse --gateway-sse build/responses-sse-captures-20260610/gateway-qwen35-mxfp8-mtp-tool-after-missing-required-args-failclosed-20260610.sse --server-log build/responses-sse-captures-20260610/direct-qwen35-mxfp8-mtp-after-missing-required-args-failclosed.server.log --gateway-log build/responses-sse-captures-20260610/gateway-qwen35-mxfp8-mtp-after-missing-required-args-failclosed.log --cache-dir build/current-responses-raw-sse-qwen35-direct-source-cache-after-missing-required-args-failclosed-20260610 --port 8898 --load-timeout-s 600 --request-timeout-s 300 --gateway-timeout-s 300 --require-reasoning-events`
- Overall artifact status: `fail`, because the classifier still includes the
  stale public tunnel capture from 2026-06-09 and that tunnel capture has
  invalid duplicate output index `0`.
- Current-source direct proof: real
  `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP` loaded as
  `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`; function args stayed exactly
  `{"value": "blue-cat"}` through argument delta/done/final response;
  reasoning lifecycle completed; final response output order is
  `[message, reasoning, function_call]`; output indices are
  `message=0`, `reasoning=1`, `function_call=2`.
- Current-source gateway proof: panel gateway live capture passed with the same
  exact function args, reasoning lifecycle, final object consistency, and
  output indices `0/1/2`. Gateway request kwargs were preserved:
  `stream=true`, `max_output_tokens=512`, `temperature=0`, `top_p=1`,
  `top_k=0`, `enable_thinking=true`, `tool_choice=required`, `tool_count=1`,
  `first_tool_name=record_fact`.
- Runtime/cache proof from server log: native MTP active and capped to D1 for
  tool requests; hybrid 10 attention + 30 SSM cache active; live TurboQuant KV
  active for attention layers; block L2 wrote four blocks / 222 tokens; the
  gateway request hit paged+hybrid cache with 222 cached tokens and clean SSM
  rederive/companion storage.
- Memory proof: before launch available memory was about `112.29 GiB`; after
  health server RSS was about `35.872 GiB` with about `76.78 GiB` still
  available.
- Still red: public tunnel parity is not cleared. The stale tunnel SSE still
  has `function_call=[0]`, `message=[0]`, no separate reasoning output item,
  and `all_present_surfaces_have_valid_output_item_indices=false`.
- Not claimed: no public tunnel rebuild/recapture, no live opencode full
  harness loop, no Gemma/MiMo/N2 clearance, and no release readiness.
- Other-agent action: rebuild/redeploy public tunnel/backend from current
  source after commit `09bfe652` or newer, then recapture the same Qwen35
  request. Keep the stale tunnel boundary red until that capture has valid
  output indices and the separate reasoning item lifecycle.

# 2026-06-10 - Qwen35 raw SSE artifact reclassified with current tunnel

- Directive check: allowed lane is Qwen/Qwen3.6/Qwen-coder
  Responses/tool/reasoning streaming parity and release-board proof accounting.
  N2 JANG_1L remains off-limits. No model launch, release, signing,
  notarization, PyPI, public download, or package action was taken.
- Correction: the live recapture runner's default tunnel path still pointed at
  the stale 2026-06-09 public tunnel SSE. The repo already had the fresh
  2026-06-10 tunnel recapture
  `build/responses-sse-captures-20260610/tunnel-qwen35-mxfp8-mtp-tool-recapture-after-strict-source-20260610.sse`.
- Action: reclassified the new direct/gateway captures from the fail-closed fix
  against the fresh tunnel capture without launching a model.
- Artifact:
  `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-missing-required-args-failclosed-20260610.json`,
  `status=pass`, `missing_captures=[]`.
- Proven: direct, gateway, and current tunnel all report same model
  `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`, preserve authoritative
  `{"value": "blue-cat"}`, have argument delta/done events, keep reasoning
  enabled without a reasoning-disable workaround, preserve final object
  consistency, and have valid output item indices. Direct/gateway use
  `message=0`, `reasoning=1`, `function_call=2`; tunnel uses `message=0`,
  `function_call=1` with no conflict.
- Checklist pointer update: `tests/cross_matrix/run_full_release_objective_checklist.py`
  now points both `RESPONSES_RAW_SSE_PARITY` and `QWEN35_RAW_SSE_PARITY` at the
  latest artifact. The guard tests were updated accordingly.
- Release tracker update:
  `docs/internal/VMLX_MLXSTUDIO_RELEASE_EXECUTION_TRACKER_2026_06_07.md`
  now names the latest Qwen raw SSE artifact and source fix commit `09bfe652`.
- Regenerated checklist:
  `build/current-full-release-objective-checklist-after-qwen-missing-required-args-failclosed-20260610.json`
  remains expected-open with `failed_count=56`, `prepackage_ready=false`, and
  `release_ready=false`.
- Verification: `.venv/bin/python tests/cross_matrix/run_responses_raw_sse_parity_contract.py ...` produced `status=pass`; `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py -k 'responses_raw_sse or qwen35_raw_sse or full_release_objective_checklist_uses_current'`
  passed `8 passed`; `python3 -m py_compile tests/cross_matrix/run_full_release_objective_checklist.py`
  passed.
- Not proven: this does not clear MiMo exactness/media, Gemma installed-app/UI
  parity, N2 JANG_1L, Step/LFM/Nemotron/DSV4 rows, full opencode harness loops,
  or release readiness.
- Other-agent action: treat Qwen35 raw SSE direct/gateway/tunnel for this
  request as green after `09bfe652` plus the latest artifact, but continue
  cross-family parser/API proof for Gemma/MiMo/N2/Qwen-coder and do not use the
  Qwen artifact as proof for other families.

# 2026-06-10 05:55 PDT - AGENTS guard updated for instruction logging and no-subagent constraint

- Request: Eric said to write every instruction/status/movement down, force the
  agent to check it, emphasize auto-tool/content/reasoning/delta/API/gateway
  parser work, and record the no Python subagent constraint "into agents.md".
- Directive check: active worktree is
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; deprecated `/Users/eric/vmlx`
  remains routing-only. N2 JANG_1L remains Eric-owned/off-limits. No release,
  signing, notarization, PyPI, upload, or model launch action was taken.
- Action: updated active `AGENTS.md` and
  `.agents/CODEX_ACTIVE_DIRECTIVES_20260610.md` to require transcribing
  current-turn user instructions/corrections into `.agents/STATUS.md` and
  `.agents/LOG.md` before substantive work. The same guard now explicitly
  prohibits Python, shell wrappers, MCP tools, or any other mechanism from
  spawning/managing subagents; direct Python remains allowed only for local
  verification, artifact inspection, proof scripts, tests, and source
  maintenance that does not prompt/supervise/summarize subagents.
- Action: updated deprecated `/Users/eric/vmlx/AGENTS.md` with the same routing
  reminder so future agents starting in the old checkout route back to the
  active worktree and carry the instruction-log/no-subagent constraints.
- Proven: documentation guard only. It records the constraint and routing
  boundary; it does not prove any model/runtime/parser/UI/release row.
- Other-agent action: before every substantive action, read active directives
  and status, write the current-turn instruction/correction down, then continue
  one live blocker at a time. Do not spawn subagents for this lane.

# 2026-06-10 05:56 PDT - Continuation objective logged before MiMo blocker work

- Request: continue toward production-quality checkpoint readiness for Nex N2
  JANGTQ2, MiMo V2.5 JANG/JANGTQ, Gemma JANG/MXFP, VL/audio/cache reuse/
  TurboQuant/reasoning/tool parser/API/gateway/UI, but do it in efficient
  phases and do not waste time building broad new test-suite infrastructure.
- Directive check: active lane is the current Python/Electron worktree
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; N2 JANG_1L remains
  Eric-owned/off-limits; no release/sign/notarize/PyPI/download action is
  allowed in this step; no subagent spawning or delegation is allowed.
- Immediate blocker being reduced: MiMo V2.5 JANGTQ_2 exactness/media runtime
  classification, using existing current proof artifacts and direct code
  inspection before any new proof.
- Must not claim: no model family is release-clear from this continuation
  unless live evidence proves the full row; no parser/JSON repair is allowed to
  mask MiMo semantic exactness; no load-only or transport-only proof is enough.

# 2026-06-10 06:02 PDT - MiMo JANGTQ2 video preprocessing cache fix live-proven, visual quality still red

- Request: reduce the MiMo V2.5 JANGTQ/JANG blocker with real fixes and live
  proof, without broad test-suite detours or fake parser/JSON semantic repair.
- Directive check: active lane was MiMo V2.5 JANGTQ_2 exactness/media/cache;
  N2 JANG_1L remained off-limits; no release/sign/notarize/PyPI/download
  action was taken; no subagents were used.
- Fix: `vmlx_engine/vision_embedding_cache.py` now stores optional
  `video_pixel_values` and `video_grid_thw` in `PixelCacheEntry`.
- Fix: `vmlx_engine/mllm_batch_generator.py` now restores those video fields on
  pixel-cache hit and stores video-only preprocessing outputs when
  `video_pixel_values` is present, instead of requiring image `pixel_values`.
- Fix: `vmlx_engine/models/mllm.py` direct/simple chat and stream paths now
  include `mimo_v2` in the video-placeholder-to-image-frame expansion gate,
  matching the fact that those direct paths pass sampled frames through
  `images=`.
- Verification: `python3 -m py_compile vmlx_engine/models/mllm.py
  vmlx_engine/mllm_batch_generator.py vmlx_engine/vision_embedding_cache.py`
  passed; focused MiMo media tests passed `26/26`; `git diff --check` passed;
  JSON artifact validated.
- Live proof: launched real
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` current-source
  server on port `8877`, with media runtime auto-enabled, visual/audio/speech
  weights bound (`364/75/20`) and native MiMo mixed full/RotatingKVCache layout.
  Repeated the same red-video `/v1/chat/completions` request twice; both
  returned HTTP 200 and health after the second request reported
  `pixel_cache_hits=1`, `pixel_cache_misses=1`, `pixel_cache_size=1`.
- Artifact:
  `build/current-mimo-v25-jangtq2-video-cache-proof-after-video-tensor-cache-fix-20260610.json`,
  `status=open`.
- Proven: video preprocessing cache preservation for video-only MiMo requests
  is no longer structurally image-only; repeated video requests hit the pixel
  cache under the real JANGTQ_2 runtime.
- Still red: the red video still answered `White.`; this does not fix MiMo
  visual semantic correctness, solid-color image correctness, text/tool/JSON
  literal exactness, fresh-process L2 restore, installed-app parity, or release
  clearance. Earlier live text exactness still mutates `MIMO-OK` to `MIMOOK`.
- Other-agent action: consume this source fix when rebuilding the app/package
  and rerun MiMo JANGTQ_2 UI video cache rows. For visual quality/exactness,
  continue artifact/logit/quant-contract diagnosis, not parser repair, sampling
  clamps, or cache/L2 chasing.

# 2026-06-10 06:05 PDT - Continuation objective logged before MiMo exactness diagnosis

- Request: continue the full checkpoint objective for N2 JANGTQ2, MiMo
  JANG/JANGTQ, Gemma JANG/MXFP, VL/audio/cache/TurboQuant/reasoning/tool
  parser/API/gateway/UI, with efficient build/fix phases and without broad new
  test-suite detours or recursive subagent behavior.
- Directive check: active lane remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; N2 JANG_1L remains
  Eric-owned/off-limits; no release/sign/notarize/PyPI/download action is
  allowed in this step; no subagents are allowed.
- Immediate blocker being reduced: MiMo V2.5 JANGTQ_2 literal/tool/JSON
  exactness classification. The current evidence says cache, parser rewrite,
  and JSON repair must not be chased as primary fixes; inspect current
  artifacts, generation/template metadata, and decode/runtime path before any
  source edit.
- Must not claim: no MiMo release clearance, no visual-quality fix, no
  exactness fix, and no installed-app parity unless live evidence proves it.

# 2026-06-10 06:09 PDT - MiMo JANGTQ2 no-fastpath exactness classifier

- Request: continue MiMo JANGTQ_2 exactness diagnosis without fake parser/JSON
  semantic repair, hidden sampling changes, broad test-suite work, subagents,
  release/signing actions, or N2 JANG_1L work.
- Evidence gathered: no-load tokenizer/template check preserves exact literals
  before generation. `AutoTokenizer` round-trips `MIMO-OK`, `blue-cat`,
  `B7-CAT-09`, `ACK-CB-742`, and exact JSON; `apply_chat_template(...,
  enable_thinking=False)` renders the exact literals into the prompt with
  `<think></think>` and no literal corruption.
- Live proof: relaunched real
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2` with
  `VMLINUX_DISABLE_MIMO_V2_COMPILED_ROUTER=1` and
  `VMLINUX_DISABLE_MIMO_V2_SWITCHGLU_FAST_PATH=1`. Startup confirmed both vMLX
  fast paths disabled while native JANGTQ/TurboQuant remained active.
- Result: exact text request still returned `MIMOOK` instead of `MIMO-OK`;
  exact JSON still returned `{"status":"ok","value":"blue","count":3}`
  instead of preserving `blue-cat`. Both requests were cold prompt misses
  (`cache_hits=0`, `cache_misses=2`), so cache-hit reuse is not the cause.
- Artifact:
  `build/current-mimo-v25-jangtq2-exactness-classifier-after-no-fastpath-live-20260610.json`,
  `status=open`.
- Classification: current MiMo JANGTQ_2 literal exactness is not caused by
  tokenizer roundtrip, chat template rendering, hidden stochastic sampling,
  parser/JSON repair, cache-hit reuse, generic TurboQuant KV, vMLX compiled
  router, or vMLX SwitchGLU decode fast path. Remaining target is native
  JANGTQ/TurboQuant codebook artifact or kernel/logit quality, especially the
  prestacked SwitchMLP routed-expert quant contract.
- Must not claim: no MiMo exactness fix, no release clearance, no installed-app
  parity, no source-vs-quant proof, and no visual/audio semantic clearance from
  this classifier.
- Other-agent action: stop chasing cache/parser/template/these vMLX fast paths
  for MiMo JANGTQ_2 exactness. Next useful work is JANGTQ_2 native
  TurboQuant/codebook logit comparison against JANG_2L or source/dequant for
  the first divergent token, or rebuilding the artifact with a corrected
  prestacked routed-expert quant contract.

# 2026-06-10 06:11 PDT - Continuation objective logged before native JANGTQ inspection

- Request: keep moving on the full checkpoint objective and avoid broad
  test-suite detours, recursive/subagent work, fake guards, parser/JSON
  semantic repair, or hidden sampling clamps.
- Directive check: active lane remains MiMo V2.5 JANGTQ_2 exactness/logit/
  artifact diagnosis in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`;
  N2 JANG_1L remains off-limits; no release/sign/notarize/PyPI/download action
  is allowed in this step.
- Immediate blocker being reduced: native JANGTQ/TurboQuant codebook runtime
  contract for MiMo prestacked SwitchMLP routed experts, because current proof
  excludes tokenizer/template/cache/generic-TQ-KV/parser/vMLX-router/
  vMLX-SwitchGLU fast paths.
- Must not claim: no MiMo exactness fix, artifact fix, installed-app parity,
  or release clearance until live output proves it.

# 2026-06-10 06:18 PDT - Post-compaction continuation before native JANGTQ inspection

- Request: continue from the MiMo JANGTQ_2 exactness classifier and do direct
  runtime proof/fix work, while keeping Eric's written constraints active.
- Directive check: active worktree remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; current lane remains MiMo
  V2.5 JANGTQ_2 exactness/logit/artifact diagnosis; N2 JANG_1L remains
  Eric-owned/off-limits; no release/sign/notarize/PyPI/download action is
  allowed in this step.
- Immediate blocker being reduced: whether the vMLX MiMo prestacked SwitchMLP
  TurboQuant binding is faithfully using the artifact's JANGTQ codebook/sign
  contract and native gather matmul shape contract.
- Must not claim: no MiMo exactness fix, no Gemma/N2/Qwen release clearance, no
  installed-app parity, and no checkpoint release readiness from source
  inspection alone.

# 2026-06-10 06:31 PDT - MiMo JANGTQ2 native TurboQuant contract classifier

- Request: continue MiMo JANGTQ_2 exactness diagnosis and avoid fake parser,
  cache, sidecar, or sampling fixes.
- Evidence gathered: the current artifact's `jangtq_runtime.safetensors`
  contains `codebook.2048.2`, `codebook.4096.2`, `signs.2048.42`, and
  `signs.4096.42`; each matches the generated `jang_tools` runtime table
  exactly (`max_abs_diff=0.0`). The main safetensor index has 141
  `.tq_packed/.tq_norms/.tq_bits` prestacked SwitchMLP groups and no indexed
  codebook/sign keys.
- Kernel proof: a direct real-tensor parity check compared
  `jang_tools.turboquant.gather_tq_matmul` against an explicit selected-expert
  dequant reference for MiMo layer 1 gate/down tensors. Broadcast gate,
  sorted-prefill gate, and per-row down shapes all matched with max absolute
  diff below `1e-6`.
- Artifact:
  `build/current-mimo-v25-jangtq2-native-tq-contract-classifier-20260610.json`,
  `status=open`.
- Classification: current MiMo JANGTQ_2 literal exactness is not explained by
  ignored sidecar codebook/sign tables, prestacked shape binding for the
  sampled groups, or native gather-kernel broadcast/sorted/per-row semantics.
  Remaining target is source-vs-quant first divergent logits or corrected
  artifact/requant profile.
- Must not claim: no exactness fix, no MiMo release clearance, no installed-app
  parity, no visual/audio/video semantic clearance, and no DMG readiness.
- Other-agent action: stop duplicating sidecar/codebook/gather-kernel checks
  for the current MiMo JANGTQ_2 artifact. Next useful work is source-vs-quant
  first divergent token/logit with the source endpoint running, or a corrected
  artifact profile such as `gate=3/up=2/down=3` or `gate=3/up=3/down=3`.

# 2026-06-10 06:38 PDT - Continuation before cross-family parser/API inspection

- Request: keep focus on auto tool usage, content deltas, reasoning deltas,
  interleaved reasoning/tool streaming, kwargs, Chat/Responses API behavior,
  gateway passthrough, output indices, and final-object consistency for coding
  harnesses such as opencode/Codex.
- Directive check: active worktree remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; N2 JANG_1L remains
  off-limits; no release/sign/notarize/PyPI/download action is allowed in this
  step; no subagents are allowed.
- Immediate blocker being reduced: inspect current Qwen/Qwen-coder empty-args
  fail-closed implementation and parser family coverage so cross-family
  tool-loop behavior does not regress to executable `{}` arguments or raw
  tool/reasoning leaks.
- Must not claim: no cross-family parser release clearance, gateway/tunnel
  clearance, installed-app parity, or release readiness unless current proof
  covers it.

# 2026-06-10 06:52 PDT - Cross-family required-argument parser fail-closed source fix

- Request: harshly focus on auto tool usage, content/reasoning/tool streaming,
  parser family behavior, and coding-harness usability without fake argument
  synthesis or reasoning disablement.
- Done: patched parser-level required-schema validation for Qwen bracket
  syntax, Kimi/Moonshot, Hunyuan/Hy3, ZAYA/Zyphra, Gemma4 native/Hermes,
  Gemma3 tool_code, and GLM-4.7 XML/JSON tool-call paths. These parser paths
  now drop missing/empty required fields instead of returning executable `{}` or
  otherwise schema-incomplete arguments.
- Source files:
  `vmlx_engine/tool_parsers/qwen_tool_parser.py`,
  `vmlx_engine/tool_parsers/kimi_tool_parser.py`,
  `vmlx_engine/tool_parsers/hunyuan_tool_parser.py`,
  `vmlx_engine/tool_parsers/zaya_tool_parser.py`,
  `vmlx_engine/tool_parsers/gemma4_tool_parser.py`,
  `vmlx_engine/tool_parsers/gemma3_tool_parser.py`,
  `vmlx_engine/tool_parsers/glm47_tool_parser.py`.
- Regression test added:
  `tests/test_tool_parser_required_args_fail_closed.py`.
- Proof artifact:
  `build/current-cross-family-tool-parser-required-args-failclosed-20260610.json`,
  `status=pass`.
- Verification: new focused test passed `7/7`; touched parser/test
  `py_compile` passed; existing touched-family parser suites passed `52/52`.
- Must not claim: this is parser-level source hardening only. It does not prove
  live same-model direct/gateway/tunnel raw SSE for every family, installed-app
  parity, Gemma media, MiMo exactness/media release clearance, N2 JANG_1L, or
  checkpoint release readiness.
- Other-agent action: consume this parser source fix in bundled Python/app
  rebuilds and continue live direct/gateway/tunnel raw SSE proof per family
  with reasoning enabled; do not add argument synthesis or semantic repair.

# 2026-06-10 07:01 PDT - Continuation before Gemma JANG/MXFP/QAT blocker work

- Request: continue the full active goal in efficient build/fix/proof blocks,
  avoid recursive/subagent behavior, avoid broad test-suite detours, and keep
  every movement written down for compaction safety.
- Directive check: active worktree remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; N2 JANG_1L remains
  off-limits; no release/sign/notarize/PyPI/download action is allowed in this
  step; no subagents are allowed.
- Immediate blocker being reduced: Gemma JANG/MXFP/QAT honest modality
  detection plus live VL/video/cache/API/UI proof gaps. Qwen35 raw-SSE for the
  reported empty-args issue is already green, and MiMo JANGTQ_2 exactness is
  classified to artifact/logit/requant profile rather than sidecar/gather/cache.
- Must not claim: no Gemma full release clearance, media clearance, installed
  app parity, checkpoint release readiness, or audio support unless current
  weight-backed live evidence proves it.

# 2026-06-10 06:29 PDT - Gemma native MXFP4 full-media proof pointer sync

- Request: put the current operating constraints into `AGENTS.md`, keep
  release focus, avoid subagents, and continue concrete Gemma/MiMo/Qwen/N2
  blocker reduction without release/sign/notarize/PyPI/download actions.
- Source change: `AGENTS.md` now explicitly records the release-focus rule,
  one-blocker-at-a-time proof rule, 128GB live-load proof target, memory/cache
  watch requirement, and the standing N2 JANG_1L off-limits boundary.
- Source change: `tests/cross_matrix/run_gemma_qat_native_mxfp4_inventory_gate.py`
  now maps existing E2B/E4B/12B native MXFP4 per-row full-media result artifacts
  into `SOURCE_FULLMEDIA_SMOKE_PROOFS`.
- Proof artifact:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-native-mxfp4-fullmedia-pointer-20260610.json`.
- Result: native MXFP4 `source_fullmedia_smoke.status=pass` for
  `gemma4_e2b_qat_native_mxfp4`, `gemma4_e4b_qat_native_mxfp4`, and
  `gemma4_12b_native_mxfp4`; each has all required labels present, no validation
  failures, Gemma4 parser/reasoning capability, exact `record_fact` args,
  image/video checks, post-media text recovery, mixed-SWA native cache,
  block-disk L2 write, and fresh-process L2 restore with `cached_tokens=56`,
  `cache_detail=paged+mixed_swa+disk`, `disk_hits=1`.
- Full checklist artifact:
  `build/current-full-release-objective-checklist-after-gemma-native-mxfp4-fullmedia-pointer-20260610.json`,
  `status=open`, `failed_count=56`.
- Verification: inventory gate `py_compile` passed; focused
  `tests/test_gemma_qat_native_mxfp4_inventory_gate.py` passed `9/9`.
- Boundary: this is a proof-pointer/source-gate sync over existing live
  artifacts, not a new heavy live run. Gemma release rows remain open for
  full live/API/UI/tunnel/installed-app proof; 26B/31B native MXFP4 full-media
  pointer rows remain missing while their narrower source live smokes pass.
  No package, signing, notarization, PyPI, download, or N2 JANG_1L action.
- Other-agent action: consume this current inventory/checklist artifact instead
  of reclassifying E2B/E4B/12B native MXFP4 full-media source smokes as missing.
  Next useful Gemma work is installed-app/UI/tunnel parity or 26B/31B native
  MXFP4 full-media proof, not rerunning the already passing E2B/E4B/12B source
  smokes.

# 2026-06-10 06:36 PDT - Continuation before Gemma 26B/31B native MXFP4 evidence check

- Request: continue the active goal in concrete build/fix/proof blocks and avoid
  recursive agent behavior or broad test-suite churn.
- Directive check: active worktree remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; N2 JANG_1L remains
  off-limits; no release/sign/notarize/PyPI/download action is allowed in this
  step; no subagents are allowed.
- Immediate blocker being reduced: Gemma 26B/31B native MXFP4 full-media proof
  gap in the current inventory gate, after E2B/E4B/12B native MXFP4 full-media
  pointers were synced.
- Must not claim: no Gemma full release clearance, UI/tunnel/installed-app
  parity, checkpoint release readiness, or audio support for 26B/31B unless
  current evidence proves it.

# 2026-06-10 06:41 PDT - Gemma 26B/31B native MXFP4 full-media proof pointer sync

- Done: inspected existing 26B/31B native MXFP4 current-source result artifacts
  under the audio-capability-gate runs and confirmed they satisfy the inventory
  full-media gate shape without advertising audio.
- Source change: `tests/cross_matrix/run_gemma_qat_native_mxfp4_inventory_gate.py`
  now maps `gemma4_26b_vl` and `gemma4_31v_or_31b_vl` to their per-row result
  artifacts:
  `build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/JANGQ_gemma-4-26B-A4B-it-qat-MXFP4/result.json`
  and
  `build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/JANGQ_gemma-4-31B-it-qat-MXFP4/result.json`.
- Proof artifact:
  `build/current-gemma-qat-native-mxfp4-local-inventory-after-26b31b-fullmedia-pointer-20260610.json`.
- Result: all five native MXFP4 rows now report
  `source_fullmedia_smoke.status=pass`; 26B/31B have all required text/tool/
  image/video labels present, no validation failures, Gemma4 parser/reasoning
  capability, exact `record_fact` args, mixed-SWA native cache, block-disk L2
  write, and fresh-process L2 restore with `cached_tokens=56`,
  `cache_detail=paged+mixed_swa+disk`, `disk_hits=1`.
- Full checklist artifact:
  `build/current-full-release-objective-checklist-after-gemma-native-mxfp4-all-fullmedia-pointer-20260610.json`,
  `status=open`, `failed_count=56`.
- Verification: inventory gate `py_compile` passed; focused
  `tests/test_gemma_qat_native_mxfp4_inventory_gate.py` passed `9/9`.
- Boundary: this closes the native MXFP4 source-fullmedia missing-pointer gap
  only. Gemma rows remain open for full live/API/UI/tunnel/installed-app proof;
  no package, signing, notarization, PyPI, download, or N2 JANG_1L action.
- Other-agent action: do not re-run Gemma native MXFP4 source-fullmedia smokes
  just to fill inventory rows. Move to installed-app/UI/tunnel parity or a
  higher-value live release blocker.

# 2026-06-10 06:48 PDT - Continuation before MiMo JANGTQ2 exactness/logit blocker work

- Request: continue the active goal in concrete build/fix/proof blocks, avoid
  broad test-suite churn, avoid recursive/subagent behavior, and keep docs
  current for compaction safety.
- Directive check: active worktree remains
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`; N2 JANG_1L remains
  off-limits; no release/sign/notarize/PyPI/download action is allowed in this
  step; no subagents are allowed.
- Immediate blocker being reduced: MiMo V2.5 JANGTQ_2 literal exactness/logit/
  artifact diagnosis after cache/parser/template/sidecar/gather-kernel causes
  were excluded by current artifacts.
- Must not claim: no MiMo exactness fix, media release clearance, installed-app
  parity, package readiness, or release readiness unless current live evidence
  proves it.

# 2026-06-10 07:06 PDT - AGENTS constraint check and current checklist refresh

- Request: write the active constraints into AGENTS.md and force this lane to
  check them instead of drifting. Checked both the deprecated wrapper
  `/Users/eric/vmlx/AGENTS.md` and this active worktree `AGENTS.md`; both
  already contain the routing guard plus the no-subagent/subagent-spawn
  constraint. This work remains in
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.
- Action: refreshed the no-heavy full release objective checklist from current
  source with
  `tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-agents-constraint-refresh-20260610.json`.
- Proof: the artifact exists and is current-source generated; `status=open`,
  `failed_count=56`, `prepackage_ready=false`, `release_ready=false`.
- Proven for MiMo JANGTQ_2 in that artifact: source media runtime wiring,
  source video/audio transport routes, dev-app Responses/tools/cache with native
  `mimo_v2_asymmetric_swa` cache, and no-source classifier boundary that keeps
  exactness as a model/artifact/logit/decode-quality blocker instead of
  parser/cache/sampling repair.
- Still not proven: MiMo release clearance, decode speed target, artifact
  exactness, media semantic quality, installed-app parity, package readiness,
  or any N2 JANG_1L claim. N2 JANG_1L remains off-limits even though the
  checklist still lists it as an open global release row.
- Other-agent action: do not rerun stale MiMo cache/parser probes. The useful
  next work is source/dequant/logit/codebook comparison or a corrected MiMo
  artifact profile, plus installed-app/UI media parity once source quality is
  worth packaging.
