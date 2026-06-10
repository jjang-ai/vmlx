# vMLX / MLXStudio release execution tracker - 2026-06-07

This is the active checklist for the Python vMLX engine and MLXStudio Electron app release objective.
It is intentionally stricter than source-only tests or narrow model smokes.

Do not use this document to justify release while any `OPEN` row remains.
Do not sign, notarize, tag, or publish downloads unless the full release checklist is green or Eric explicitly overrides the lock in the current turn.

## Current authoritative artifacts

| Surface | Artifact | Current result |
| --- | --- | --- |
| Full release checklist | `build/current-full-release-objective-checklist-after-agents-constraint-refresh-20260610.json` | `status=open`, `failed_count=56`; consumes current Qwen35 same-model direct/gateway/tunnel Responses raw-SSE parity, Qwen empty-args fail-closed source boundary, refreshed Gemma 12B JANG_4M exact-code proof, Gemma QAT/native MXFP4 inventory with all five native MXFP4 source-fullmedia rows now mapped, current MiMo/N2 objective details, MiniMax current-source smoke boundary, installed-app parity, and package readiness gates. MiMo source media routing, source video/audio transport, and dev-app Responses/tools/cache are green in this board, while MiMo exactness, decode speed, media semantic quality, installed-app parity, and release clearance remain open. The stricter N2 JANGTQ2 long-delta proof exposed a Responses required-tool stream fail-closed bug that is now fixed in source and live-rerun: the second turn now surfaces `tool_calls_required` without persisting `!` output or a fake assistant placeholder, but the model still does not complete that stricter second required-tool request. This is still release-red; no signing, notarization, tagging, or download publishing. |
| Installed-app runtime parity | `build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json`; `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json` | Runtime parity is now `status=pass` after rebuilding and installing `/Applications/vMLX.app` from current source. Bundled Python imports `vmlx_engine 1.5.56`, `mlx 0.31.2`, `mlx-lm 0.31.3`, and `mlx-vlm 0.5.0`, and the installed bundle has source hash parity for the tracked engine/runtime/cache/parser/model-register files. Panel settings contract remains `status=pass`, including parser/reasoning settings, model-owned generation defaults, max output/context split, cache flags, and CLI flag registration. This is installed-app parity only; it does not clear live per-family model/UI rows or Developer ID signing/notarization. |
| Packaged integrity / signing preflight | `build/current-packaged-integrity-contract-after-installed-app-rebuild-20260606.json`; `build/current-signed-checkpoint-dmg-readiness-20260609.json` | The older packaged-integrity artifact still records the previous `packaged_app_developer_id_signing_blocked` state, but the current signed-checkpoint audit supersedes that keychain status: fresh Developer ID signing is `pass`, notary history access is `pass`, and `vmlx-build.keychain-db` reports `no-timeout`. Local `/Applications/vMLX.app` from `panel/scripts/build-and-install.sh` remains ad-hoc signed and valid on disk; it is not a signed/notarized checkpoint DMG. |
| Signed checkpoint DMG readiness | `build/current-signed-checkpoint-dmg-readiness-20260609.json` | `status=open`. Existing local June 5 `vMLX-1.5.56-{sequoia,tahoe}-arm64.dmg` files are Developer ID signed, stapled, and Gatekeeper accepted, but they are not current-source checkpoint proof for HEAD `d1054f41`. After following `/Users/eric/wiki/infra/apple-notarization.md`, fresh Developer ID signing with `Developer ID Application: ShieldStack LLC (55KGF2S5AY)` is `pass`, notary profile access with `--keychain ~/Library/Keychains/vmlx-build.keychain-db --keychain-profile vmlx-notary` is `pass`, and required next steps are only `rebuild_current_source_dmg_flavors` plus `notarize_staple_and_verify_current_dmgs`. |
| Current DMG build prepackage gate | `build/current-release-regression-manifest-pre-dmg-release-build-after-keychain-unlock-20260609.json` | Running `VMLINUX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-pre-dmg-release-build-after-keychain-unlock-20260609.json panel/scripts/build-release-dmgs.sh sequoia` stopped at the official `--require-prepackage-ready` gate before building a DMG. Result: `status=fail`, `prepackage_ready=false`, `release_ready=false`, and `current_proof_sweep.component_ok.packaged_app_developer_id_signing=true`. The live blockers are model/API/UI/proof rows such as MiMo JANG_2L runtime quality, MiniMax #179 reporter parity, Gemma26 installed-app memory stress, real UI matrix, N2 JANG_1L, Responses raw SSE, and DSV4, not Apple signing access. |
| Cross-model tool parser package parity coverage | source patch: `panel/scripts/verify-bundled-python.sh`, `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_packaged_integrity_contract.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py` | Source/package-gate green. Every top-level `vmlx_engine/tool_parsers/*.py` file is now included in bundled-python, release-gate, staged packaged-integrity, and installed-app parity hash surfaces. This protects Qwen empty-arg fail-closed behavior, XML-function parsing, Gemma4 native calls, MiniMax bare invoke parsing, DSML, and the rest of the registered parser matrix from stale packaged drift. Focused package/hash tests failed before wiring and passed after. This is package/parity coverage only, not same-model tunnel raw-SSE or live UI/release clearance. |
| Runtime patch package parity coverage | source patch: `panel/scripts/verify-bundled-python.sh`, `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py`, `tests/cross_matrix/run_packaged_integrity_contract.py`, `tests/cross_matrix/run_current_regression_suite.py` | Source/package-gate green. Every auto-installed `vmlx_engine.runtime_patches` module is now covered by current-suite source hashes and package parity hash surfaces: DSV4 registration, Gemma4 tiny-RGB processing, Gemma4 mixed pixel-values vision coercion, Kimi fp32 MLA patch installer, and MLX/MLX-VLM compatibility shims. Focused tests passed `6/6` after red/green wiring. This is package/source-hash coverage only, not live Gemma/Kimi/DSV4 release clearance. |
| Gemma4 vision runtime-bootstrap guard | source guard: `tests/test_vl_video_regression.py::TestIssueGuards::test_mlxstudio_88_gemma4_vision_pixel_values_list_coercion` | Source/test green. The mlxstudio#88 guard now imports `vmlx_engine` before inspecting `mlx_vlm.models.gemma4.vision`, matching the current runtime-patch bootstrap path instead of requiring raw upstream `mlx_vlm` source mutation. It asserts the per-item `mx.array` coercion and `_vmlx_gemma4_pixel_values_patch` marker after bootstrap. This prevents false negative Gemma media guards while keeping package parity and full Gemma media/UI/tunnel release rows open. |
| Qwen/N2 native-MTP package parity coverage | source patch: `panel/scripts/verify-bundled-python.sh`, `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py`, `tests/cross_matrix/run_packaged_integrity_contract.py` | Source/package-gate green. `native_mtp.py` and `patches/mlx_lm_mtp/{__init__.py,batch_generator.py,cache_rollback.py,deepseek_v4_model.py,qwen35_model.py}` are now covered by bundled-python, release-gate, packaged-integrity, and installed-app parity hash surfaces, matching the already covered `patches/mlx_vlm_mtp/qwen35_vl.py`. This protects Qwen/N2 native-MTP detection, GatedDelta patching, rollback state, and BatchGenerator draft/verify dispatch from packaged drift. Focused package/parity tests passed `4/4` after red/green wiring. This is not live N2/Qwen cache/API/UI release clearance. |
| TQ-native disk cache package parity coverage | source patch: `panel/scripts/verify-bundled-python.sh`, `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py`, `tests/cross_matrix/run_packaged_integrity_contract.py` | Source/package-gate green. `tq_disk_store.py` and `cache_record_validator.py` are now covered by bundled-python, release-gate, packaged-integrity, and installed-app parity hash surfaces so stale packages cannot silently drift on compressed `TurboQuantKVCache` disk encode/decode or unsafe cache-restore validation. Focused package/parity tests passed `4/4` after red/green wiring. This is package/parity coverage only, not live N2/MiMo/Gemma cache/UI release clearance. |
| Qwen/N2 hybrid TurboQuant package parity coverage | source patch: `panel/scripts/verify-bundled-python.sh`, `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py`, `tests/cross_matrix/run_packaged_integrity_contract.py` | Source/package-gate green. `utils/hybrid_tq_cache.py` is now covered by bundled-python, release-gate, packaged-integrity, and installed-app parity hash surfaces so stale packages cannot silently drift on Qwen3.6/N2 selective live TurboQuant KV for attention layers versus SSM companion caches. Focused package/parity tests passed `6/6` after red/green wiring. This is package/parity coverage only, not live N2 cache/tool/UI release clearance. |
| Gemma4 Unified packaged parity hash coverage | source patch: `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py`, `tests/cross_matrix/run_packaged_integrity_contract.py` | Source/package-gate green. `models/gemma4_unified_register.py` and vendored `models/gemma4_unified/*` are now included in release-gate, installed-app parity, and packaged-integrity hash surfaces, matching the existing bundled-python import gate. This closes a silent packaged omission path for `mlx_vlm.models.gemma4_unified`; it does not rebuild the app or clear installed-app Gemma media/cache/UI/tunnel rows. |
| Single-active cache max_kv_size hybrid guard | source patch: `vmlx_engine/utils/single_batch_generator.py`; tests: `tests/test_single_active_batch_generator.py` | Source/no-heavy green. After checking upstream `ml-explore/mlx-lm` PR #1343, vMLX now keeps generic `max_kv_size` from converting Gemma4/Gemma4 Unified, MiMo V2, Qwen3.6/N2, explicit hybrid cache, hybrid/mixed/SWA/SSM/Mamba subtypes, or mixed sliding+full/global layer configs into bounded rotating KV windows. Plain KV models still receive explicit `max_kv_size`. This protects cache semantics; it is not live N2/MiMo/Gemma release clearance. |
| Gemma4 shared-KV MLX-format compatibility | source patch: `vmlx_engine/runtime_patches/mlx_vlm_compat.py`; tests: `tests/test_mlx_lm_runtime_patches.py`, `tests/test_engine_audit.py` | Source/no-heavy green. The Gemma4 shared-KV runtime patch now drops materialized shared-layer K/V weights in both sanitize and `load_weights()` paths, covering `language_model.model.layers.*` and `model.language_model.model.layers.*` MLX key layouts. The bundled-python hash gate test now explicitly covers the runtime patch files needed by this path. This prevents a rebuilt package from silently omitting the compatibility shim, but it is not Gemma QAT live media/cache/UI/installed-app release clearance. |
| Objective proof digest | `build/current-objective-proof-after-n2-jangtq2-l2-live-proof-20260609.json` | status remains open; the N2 row consumes current-source N2 JANGTQ2 chat/cache/Responses proof and fresh-process L2 restart proof. MiMo objective evidence uses current present artifacts only. Gemma QAT details now list the five exact source-smoke summary artifacts plus media-backing facts, while keeping full live proof red. N2 still requires JANG_1L runtime/cache/API/UI proof, media/UI/installed-app parity, same-model tunnel parity, and release evidence; MiMo and Gemma still require their full live/UI/release proof. |
| Release regression manifest | `build/current-release-regression-manifest-after-mimo-media-runtime-stamp-gate-20260608.json` | `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; MiMo root-cause evidence now includes the media runtime stamp gate, stale-cache cleanup, no-source exactness classifier, object image/video E2E, audio waveform E2E pass, and L2 restart restore. |
| MiMo current audit | `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`; classifier `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json` | `status=open`, `local_release_clearance=false`; release/objective/checklist defaults now consume the cache-vs-nocache audit plus lossless-token-trace classifier. Current positives include manifest/structural integrity, text-cache narrow proof, SwitchGLU parity, cache-vs-no-cache next-token proof, tool protocol, API/cache contract, and JANGTQ2 L2 restore. Open rows remain JANGTQ2 artifact exactness, strict decode speed, VL/audio/video wiring/E2E, JANGTQ2/JANG_2L media/L2, UI, installed-app, and package parity. |
| MiMo vision head-dim source parity | `build/current-mimo-v25-vision-head-dim-source-parity-20260610.json` | `status=pass`; vMLX now matches upstream MiMo vision runtime's missing-`qk_channels` default of `64` instead of using `hidden_size / num_heads` (`40` for the current 1280/32 bundle). Focused MiMo media runtime tests passed and a direct Torch-vs-MLX first-block probe with real `visual.blocks.0.*` weights reported `vision_head_dim=64`, `block_head_dim=64`, and `mean_abs_diff=0.000536009669303894`. This is source-parity hardening only; MiMo JANGTQ_2 visual semantic correctness, literal exactness, installed-app parity, and release clearance remain open until fresh live proof passes. |
| MiMo JANGTQ2 live media/tools/cache after head-dim fix | `build/current-mimo-v25-jangtq2-live-media-tools-cache-after-head-dim-20260610.json` | `status=open`; current source loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`, bound preserved visual/audio/speech weights, returned `vMLX` for the app icon image, emitted valid Chat tool-call structure, emitted valid Responses tool-call streaming deltas/done with output indices `message=0` and `function_call=1`, and proved same-process paged native MiMo mixed-SWA cache reuse (`cache_hit_tokens=280`, last Responses request `cached_tokens=253`). Still red: literal exactness (`MIMO-OK` -> `MIMOOK`, `blue-cat` -> `blue cat`), red video semantics (`White.`), audio exactness/hygiene, block-disk L2/fresh-process restore, UI/installed-app parity, and release clearance. |
| MiMo JANGTQ2 dev-app image route refresh | `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-image-after-source-media-detect-20260610-proof.json`; `docs/internal/agent-notes/current-real-ui-dev-app-mimo-v25-jangtq2-red128-image-after-source-media-detect-20260610-proof.json`; `docs/internal/agent-notes/current-mimo-v25-jangtq2-direct-color-ab-20260610.md` | `status=open`; the stale dev/installed artifacts that logged `tier=mimo_v2_preserved_text_runtime result=False` are no longer current for source. Fresh source/dev-app runs with `--is-mllm` load `model_type=mllm`, log `engine_is_mllm=true`, auto-enable the preserved MiMo media runtime, bind `visual=364`, `audio_encoder=75`, `speech_embeddings=20`, process one image (`num_images_processed=1`), persist the image attachment, skip media prompt cache store, and write block-disk L2 for the text turn. Still red: both the default 1x1 red fixture and a 128x128 solid red image answer `Blue.` in the UI proof. A same-process direct API classifier with the compiled router disabled returned `Blue.` for text-only/no image and `White.` for red, green, blue, white, and black solid-image requests, so `vl_image` semantic proof remains failed and this is not a red-only channel swap. Boundary: do not keep chasing the old force-MLLM override; next MiMo media work is Torch-vs-MLX visual/logit/splice comparison or artifact/runtime quality, plus installed-app bundled parity after source rows are release-worthy. |
| MiMo JANGTQ2 video preprocessing cache fix | `build/current-mimo-v25-jangtq2-video-cache-proof-after-video-tensor-cache-fix-20260610.json`; source patch: `vmlx_engine/vision_embedding_cache.py`, `vmlx_engine/mllm_batch_generator.py`, `vmlx_engine/models/mllm.py` | `status=open`; source now preserves `video_pixel_values` and `video_grid_thw` in the MLLM pixel cache and stores video-only processor outputs instead of treating the cache as image-only. Direct/simple MiMo chat and stream paths also expand sampled video frames to image placeholders when they route through `images=`. Live current-source repeated-video probe loaded the real JANGTQ_2 bundle, returned HTTP 200 twice, and health reported `pixel_cache_hits=1`, `pixel_cache_misses=1`, `pixel_cache_size=1`. Boundary: this proves video preprocessing cache preservation, not visual semantic correctness; the red fixture still answered `White.`, and MiMo exact literals remain red. |
| MiMo JANGTQ2 no-fastpath exactness classifier | `build/current-mimo-v25-jangtq2-exactness-classifier-after-no-fastpath-live-20260610.json` | `status=open`; tokenizer roundtrip and chat-template rendering preserve `MIMO-OK`, `blue-cat`, `B7-CAT-09`, `ACK-CB-742`, and exact JSON before generation. A live current-source run with `VMLINUX_DISABLE_MIMO_V2_COMPILED_ROUTER=1` and `VMLINUX_DISABLE_MIMO_V2_SWITCHGLU_FAST_PATH=1` confirmed the vMLX compiled router and vMLX SwitchGLU decode fast path were disabled, yet exact text still returned `MIMOOK` and exact JSON returned `{"status":"ok","value":"blue","count":3}`. This excludes tokenizer/template, parser/JSON repair, hidden sampling, cache-hit reuse, generic TQ KV, and these two vMLX fast paths as primary causes. Remaining target is native JANGTQ/TurboQuant codebook artifact or kernel/logit quality, especially prestacked SwitchMLP routed experts. |
| MiMo JANGTQ2 native TurboQuant contract classifier | `build/current-mimo-v25-jangtq2-native-tq-contract-classifier-20260610.json`; `build/current-mimo-v25-jangtq2-native-tq-allproj-contract-20260610.json` | `status=open`; current artifact sidecar tables `codebook.2048.2`, `codebook.4096.2`, `signs.2048.42`, and `signs.4096.42` exactly match the generated `jang_tools` runtime tables, so generated-table use is equivalent to sidecar hydration for this bundle. The main index has 141 prestacked `.tq_packed/.tq_norms/.tq_bits` SwitchMLP groups and no indexed codebook/sign tensors. The first direct real-tensor parity check compared native `gather_tq_matmul` against explicit selected-expert dequant reference for MiMo layer 1 gate/down tensors; broadcast, sorted, and per-row shape contracts matched below `1e-6` max absolute diff. The all-projection refresh now covers layers `1`, `2`, `23`, and `47`, projections `gate_proj`, `up_proj`, and `down_proj`, real experts `[0, 3, 17, 251]`, and broadcast/sorted/per-row contracts with max abs diff `1.4901161193847656e-08`. Boundary: this narrows the exactness blocker away from sidecar/gather-kernel/projection shape bugs, but MiMo JANGTQ_2 literal exactness remains artifact/logit/requant-profile red. |
| MiMo no-source exactness classifier | `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json` | `status=open`; primary classification remains `jangtq2_plain_literal_copy_fails_before_parser_or_json_repair`. The classifier now consumes `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json`, where no-cache, warm-store, and paged cache-hit rows all return `ACK` with matching top-10 logprob distribution and `cached_tokens=31`; cache/KV/L2, parser argument rewrite, and hidden stochastic sampling are excluded as primary causes. The remaining blocker is model/artifact/logit exactness: current JANGTQ_2 still mutates literal tool/JSON/plain values before parser or JSON repair can matter. Source-vs-quant remains missing by RAM policy. |
| MiMo JANG_2L runtime thinking-off proof | `build/current-mimo-v25-jang2l-json-sentinel-token-trace-20260608/summary.json`; `build/current-mimo-v25-jang2l-json-sentinel-after-enable-thinking-forward-20260608/summary.json`; `build/current-mimo-v25-jang2l-json-sentinel-toplogits-trace-scores-20260608/summary.json` | Runtime bug fixed partial. Before the forwarding fix, token trace showed the non-streaming batched MLLM request had `enable_thinking=None`, sampled `<|im_end|>` as the first token, and stopped with empty content. After forwarding `enable_thinking=false` into `MLLMScheduler.generate`, trace shows MiMo thinking/EOS processors active, first token `{\"`, final stop on `<|im_end|>`, and visible JSON output. The top-logit trace proves the remaining failure is not stop-token handling or JSON repair: the model/runtime score ranked token `1` above requested token `3` at the `count` position (`22.05` vs `20.80`), producing `{\"status\":\"ok\",\"value\":\"B7-CAT-09\",\"count\":1}` vs expected `count:3`. |
| MiniMax #179 current audit | `build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json` | `status=open`; local source/installed app diagnostics and cancel route are clean, local metadata fallback proves the local MiniMax K artifact shape, and the audit now consumes current-source MiniMax Small JANGTQ smoke `build/current-all-local-model-smoke-minimax-small-jangtq-cache-language-after-bare-invoke-tool-20260609/summary.json` with tool/reasoning/parser/cache/L2/TQ checks green. This is deliberately scoped source evidence only: reporter parity metadata, reporter generation-config/sampling parity, reporter server hash drift, reporter-side session/log/cancel lifecycle proof, and reporter-machine same-prompt raw SSE/visible/reasoning capture are still absent. |
| LFM real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json` | `status=pass`; `lfm25-mxfp4-responses-tools-cachecontrols-20260607` refresh clears LFM mixed-identity matrix partial |
| Qwen 3.6 27B MTP real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; current model identity, Responses streaming, built-in tool loop, reasoning display, image answer `Red.`, settings overrides, MTP-compatible deterministic sampling, native MTP activation, hybrid SSM cache, TurboQuant attention KV, L2 block disk, SSM companion disk, and server cache controls are proven. The same combined reasoning/tools/image row failed at `max_tokens=96`, so thinking-mode UI proofs need a realistic output budget. |
| Qwen 3.6 27B MTP installed-app current proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; installed `/Applications/vMLX.app` bundle used bundled Python `vmlx_engine 1.5.56`, Responses streaming, tool loop, reasoning display, image answer, native MTP, hybrid SSM cache, TurboQuant attention KV, block L2, SSM L2, server cache controls, settings persistence, parser/language leak checks, and installed-app UI route are proven. Functional parity is green for this slice, but packaged Metal acceleration symbols reported `nax_symbols=0` / `naxtile_symbols=0`, so installed-app speed/perf parity remains a release risk. |
| Qwen 3.6 27B MTP restart-L2 restore proof | `build/current-qwen27-mxfp4-mtp-restart-l2-restore-20260609/summary.json` | `status=pass`; phase 1 writes block L2 and SSM companion disk state, phase 2 fresh-process restore returns visible ack with `cache_detail=paged+ssm+disk`, block disk hit, typed `hybrid_ssm_v1`, attention-only TurboQuant KV, native SSM companion policy, and native MTP active. |
| Qwen 3.6 27B MTP installed-app video proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json` | `status=pass`; installed app video needed `max_tokens=512`. The max-256 attempt decoded video but stopped reasoning-only with no visible answer. The passing max-512 proof decoded the MP4 data URL, extracted six frames, returned visible video content, and proved `video_where_supported`, reasoning display, Responses streaming/cache detail, tools, native MTP, typed `hybrid_ssm_v1`, TurboQuant attention KV, block L2, SSM L2, server cache controls, and media-safe skip of media prompt cache store. |
| Qwen 3.6 27B MTP long-context cache-tail proof | `build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json` | `status=pass`; installed bundled Python served a 31,647-input-token prompt. Cold request returned exact begin/middle/end tail markers and wrote 495 block-L2 entries (`31,646` tokens) plus SSM companion disk state (`63,262` tokens). Warm request returned exact markers again in 9.5s with `cached_tokens=31646`, `cache_detail=paged+ssm`, 1,485 block-disk hits, typed `hybrid_ssm_v1`, TurboQuant attention KV, and native MTP active. |
| Qwen 3.6 35B MXFP8 MTP real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; current model identity, Responses streaming, built-in tool loop, reasoning display, image answer, deterministic UI sampling, native MTP activation under tool-compatible D1, trained top-k 8, hybrid SSM cache, TurboQuant attention KV, block-disk L2, SSM companion disk, and server cache controls are proven |
| Qwen 3.6 35B MXFP8 MTP installed-app current proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-35b-mxfp8-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json` | `status=pass`; installed `/Applications/vMLX.app` bundle used bundled Python `vmlx_engine 1.5.56`, Responses streaming, tool loop, reasoning display, image answer, native MTP, trained top-k 8, hybrid SSM cache, TurboQuant attention KV, block L2, SSM L2, server cache controls, settings persistence, parser/language leak checks, and installed-app UI route are proven. Image media was actually exercised in the corrected run (`requestedMedia=true`, `num_images_processed=1`, `vl_image` present). Functional speed was strong in this slice (`83-89 live tok/s`), but packaged Metal acceleration symbols still report `nax_symbols=0` / `naxtile_symbols=0`, so packaged acceleration parity remains a release risk. |
| Qwen 3.6 35B MXFP8 MTP restart-L2 restore proof | `build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json` | `status=pass`; bundled app Python served two fresh processes against the same block cache. Phase 1 wrote 27 block-L2 entries and SSM companion disk state. Phase 2 returned visible `ACK-QWEN35-L2`, restored `cached_tokens=1695` with `cache_detail=paged+ssm+disk`, hit 27 block-disk blocks, hit SSM companion disk once, exposed typed `hybrid_ssm_v1`, attention-only TurboQuant KV, and native MTP depth 3 active. |
| Qwen 3.6 35B MXFP8 MTP installed-app video proof | `docs/internal/agent-notes/current-real-ui-installed-app-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json` | `status=pass`; installed app video at `max_tokens=512` decoded the MP4 data URL, extracted six frames, returned visible video content, and proved `video_where_supported`, reasoning display, Responses streaming/cache detail, tools, native MTP depth 3, trained top-k 8, typed `hybrid_ssm_v1`, TurboQuant attention KV, block L2, SSM L2, server cache controls, and media-safe skip of media prompt cache store. |
| Qwen 3.6 video UI current proofs | `docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max256-20260607-proof.json`; `docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json` | `status=pass` for both current Qwen MTP bundles; video data URL persisted, six frames decoded, `video_where_supported`, reasoning display, tool loop, native MTP, typed hybrid SSM cache, TurboQuant attention KV, block L2, SSM L2, and server cache controls are proven. Qwen35 needed `max_tokens=512`; at 256 it decoded video but stopped reasoning-only. |
| N2 JANGTQ2 stricter required-tool fail-closed proof | `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-prevresp-longdelta-after-required-failclosed-no-placeholder-20260610-proof.json`; source patches: `vmlx_engine/server.py`, `panel/src/main/ipc/chat.ts` | `status=fail` by release assertion, but the previous bad behavior is fixed. Current Electron dev app loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`, completed the first required `run_command` turn, then the stricter second required-tool request failed with renderer send error carrying `tool_calls_required`. The server emitted no visible second-turn content, the UI persisted no `!` output and no fake second assistant placeholder (`assistantCount=1`, `messageCount=3`), and cache telemetry still showed N2 typed `hybrid_ssm_v1`, live attention TurboQuant KV, paged+SSM cache hit `384`, block L2 `3405` tokens, and SSM companion disk `9421` tokens. Still red: the model did not create `real_ui_tool_probe_2.txt` or complete the stricter second tool loop, so this is fail-closed/error-surface proof, not long-tool-loop release clearance. |
| Gemma 4 12B JANG_4M real UI current proof | `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-responses-tools-image-cachecontrols-after-media-fallback-20260607-proof.json` | `status=pass`; default optimized server launch, Responses streaming, built-in tool loop, image response `Red`, mixed-SWA paged cache hits, block-disk L2 writes, cache controls, settings persistence, and parser/language leak checks are proven |
| Gemma 4 12B #191 source startup proof | `build/current-gemma4-12b-issue191-source-startup-visible-proof-20260609.json` | `status=pass`; current source imports the Gemma4 unified assistant alias, serves `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`, returns visible `GEMMA4-OK`, finishes `stop`, and stays healthy. The full release checklist now consumes this as `gemma4_12b_issue191_startup_*` green rows. This does not clear the older JANG_4M tools/cache nomedia artifact, media smoke, QAT/native MXFP4, UI/installed-app/tunnel, or release rows. |
| Gemma 4 12B MXFP4 source image proof | `build/current-gemma4-12b-mxfp4-media-smoke-after-release-doc-correction-20260609.json` | `status=pass`; current source served `/Users/eric/models/OsaurusAI/gemma-4-12B-it-MXFP4` as `gemma4-12b-mxfp4-media-smoke`, reported vision capability, processed a red PNG data URL through `/v1/chat/completions`, and returned visible `Red` with `no_channel_leak=true`. Health records MXFP4/JANG dispatch (`weight_format=mxfp4`, `mlx_affine_quantized_matmul`, `metal_na_active_on_host=true`). This is source API image proof only; it does not clear Gemma tools/cache/UI/installed-app/tunnel/audio/video release rows. |
| PyPI package checkpoint | `https://pypi.org/project/jang/2.5.30/`; `https://pypi.org/project/vmlx/1.5.56/` | Published `jang==2.5.30` and `vmlx==1.5.56` from fresh `/tmp` build artifacts after `twine check` passed for both wheels and sdists. Clean temp venv no-deps install from PyPI succeeded for both packages, and installed `vmlx` metadata requires `jang>=2.5.30` for the hard dependency plus `mxtq`/`jang` extras. This is a PyPI package checkpoint only, not a signed/notarized DMG release. |
| Gemma QAT/native MXFP4 expansion | OPEN | Newly tracked release scope: Gemma4 E2B/E4B QAT/native MXFP4, Gemma4 12B native MXFP4/QAT-style bundles, and Gemma4 26B/31V VL/video bundles if present. Existing Gemma4 JANG_4M image/cache proof does not cover these rows. Required proof includes model-owned generation defaults, parser/tool/reasoning behavior, visual/audio/video where advertised, mixed-SWA/native cache telemetry, TurboQuant KV boundaries where valid, block-disk L2 write and fresh-process restore, Responses streaming content/tool args, UI/CLI settings parity, and installed-app startup parity. |
| Gemma4 shared-KV mlx-format load compatibility | `vmlx_engine/runtime_patches/mlx_vlm_compat.py`; `tests/test_mlx_lm_runtime_patches.py` | Source/test green after adapting upstream `Blaizzy/mlx-vlm` PR #1336. The runtime patch now filters materialized shared-layer k/v tensors not only in `sanitize`, but also in strict `Model.load_weights`, covering mlx-format checkpoints where upstream loading can skip sanitize. Regression `test_mlx_vlm_gemma4_shared_kv_load_weights_drops_mlx_format_materialized_kv` passes. This is no-heavy load compatibility, not live Gemma media/UI/tunnel/installed-app release proof. |
| Gemma QAT/native MXFP4 E2B/E4B audio/video/tools/cache proof | `build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`; `build/current-all-local-model-smoke-gemma4-e4b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json` | `status=pass`, `failures=0` for both source smokes after the harness quoted the exact tool-result target string instead of leaving the final period ambiguous. E2B/E4B pass audio transcription (`Blue`/`blue`), image, video, post-audio text recovery, required tool use, tool-result continuation, mixed-SWA cache telemetry, and block-disk L2 restart restore (`disk_hits=1`). This is current-source API proof, not installed-app/UI/tunnel release clearance. |
| Gemma QAT/native MXFP4 12B/26B/31B audio capability boundary | `build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`; `build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json`; `build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json` | `status=pass`, `failures=0`/`failed=0` for 12B, 26B, and 31B source smokes covering text, image, video, tools, cache, and L2. Runtime capabilities now gate Gemma4 audio from real `audio_config` plus `audio_tower.*` weights instead of token metadata; 26B/31B advertise `text/vision/video` with `audio: not_advertised`. Index audit shows E2B/E4B have full audio tower weights, 26B/31B have zero audio weights, and 12B has only `embed_audio.embedding_projection.weight`; audio must remain capability-gated honestly per bundle. |
| Gemma local QAT/native MXFP4 inventory gate | `build/current-gemma-qat-native-mxfp4-local-inventory-after-12b-installed-app-ui-proof-20260610.json` | `status=open`, `count=21`, `missing_required_rows=[]`, `source_live_smoke_open_rows=[]`; the no-heavy gate finds local Gemma4 E2B/E4B/12B/26B/31B QAT/native MXFP4 rows and records current-source smoke artifacts for all five required rows. All five native MXFP4 rows now have `source_fullmedia_smoke.status=pass` from per-row result artifacts: E2B/E4B/12B from the fullmedia-tools-L2 runs and 26B/31B from the audio-capability-gated tools/image/video/L2 runs. E2B, E4B, and 12B QAT JANG_4M now also have installed-app UI/API/cache proof from `/Applications/vMLX.app` using bundled Python, Responses wire API, real built-in tool loops, reasoning display, mixed-SWA native cache telemetry, and block-disk L2 tokens on disk (`gemma4_e2b_qat_jang4m_installed_app_ui_api_cache_proven=true`, `gemma4_e4b_qat_jang4m_installed_app_ui_api_cache_proven=true`, `gemma4_12b_qat_jang4m_installed_app_ui_api_cache_proven=true`). The 12B UI proof has visible first/second assistant answers and cache hits, but also records a post-answer reasoning-only auto-continue warning; treat that as a UI polish/open-boundary item, not as release clearance. The 26B installed-app attempt is intentionally not registered green: `docs/internal/agent-notes/current-real-ui-installed-app-gemma4-26b-qat-jang4m-responses-tools-cachecontrols-visible-chat-20260610-proof.json` and its screenshot show bundled-Python load/cache success but twelve duplicate `run_command` executions on the second turn, so 26B remains open for agentic-loop cleanup. Each row has all required source labels present, no validation failures, Gemma4 parser/reasoning capability, exact `record_fact` tool args, image/video checks, post-media text recovery, mixed-SWA native cache, block-disk L2 write, and fresh-process L2 restore (`cached_tokens=56`, `cache_detail=paged+mixed_swa+disk`, `disk_hits=1`). 12B remains honestly audio-gated because it has config/token audio metadata but no `audio_tower.*` weights; 26B/31B advertise text/vision/video and no audio. The release rows still remain `open` until remaining live/API/UI/tunnel/installed-app proofs exist for all required rows. |
| Gemma4 12B JANG_4M no-media exact-code refresh | `build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-code-column-prompt-20260610/JANGQ_gemma-4-12B-it-JANG_4M/result.json` | `status=pass`, `failures=[]`; the old exact-code prompt let the model indent the third top-level line by one space. Live cache-disabled isolation proved the clearer prompt, explicitly requiring the third line to start at column 1, returns exact `def add(a, b):\n    return a + b\nprint(add(2, 3))`. The smoke prompt now uses that contract and both matching Gemma 12B JANG_4M no-media rows passed. This is a prompt-contract fix for the strict exact-code row, not parser repair or output rewriting. |
| API/cache/Responses no-heavy contract | `build/current-responses-required-tool-stream-fail-closed-after-n2-longdelta-20260610.json`; `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json` | `status=pass` for the current stricter source guard. Responses streaming with `tool_choice=required` now buffers visible content until a real tool call exists; if the model emits a visible preamble plus an empty XML function missing required args, the parser drops the call, emits no argument delta/done events, suppresses visible text deltas, returns `tool_calls_required`, and completes the Responses object as `status=failed` with empty `output_text`/`output`. Panel IPC now rethrows server SSE error events instead of swallowing them as malformed-line warnings, and token-count-only failed responses no longer persist fake `[Generation interrupted]` assistant messages. Existing no-heavy coverage still protects buffered args, reasoning-channel args, tool-only output index ordering, and no `{}` argument emission. This is source/no-heavy plus dev-app fail-closed proof, not installed-app, tunnel, or release clearance. |
| Responses raw SSE local/gateway/tunnel parity | `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-missing-required-args-failclosed-20260610.json`; source fix `09bfe652` | `status=pass` for the current reported Qwen/Qwen-coder class: same-model direct server, local panel gateway, and current public tunnel all capture `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`, preserve authoritative `record_fact` args `{"value": "blue-cat"}`, emit argument delta/done events, keep reasoning enabled, report required reasoning events, preserve final object consistency, and use valid output item indices. Current source also rejects schema-required empty XML/JSON tool arguments in Qwen/XMLFunction/MiMo/Nemotron/Step parser paths instead of emitting executable `{}` args. This clears the Qwen raw-SSE parity row only; Gemma4 E2B tunnel availability remains a separate deployment/model-advertisement gap, not the active Qwen empty-args report. |
| Cross-family parser required-argument fail-closed hardening | `build/current-cross-family-tool-parser-required-args-failclosed-20260610.json`; source patch: `vmlx_engine/tool_parsers/{qwen,kimi,hunyuan,zaya,gemma4,gemma3,glm47}_tool_parser.py` | `status=pass` for parser-level source contract. Qwen bracket, Kimi/Moonshot, Hunyuan/Hy3, ZAYA/Zyphra, Gemma4 native/Hermes, Gemma3 tool_code, and GLM-4.7 XML/JSON parser paths now apply the shared required-schema helper and drop missing/empty required arguments instead of returning executable `{}`/schema-incomplete calls. Verification passed: new focused test `7/7`, touched parser `py_compile`, and existing touched-family parser suites `52/52`. Boundary: this is not live direct/gateway/tunnel proof for every family and does not clear installed-app, Gemma media, MiMo exactness/media, N2 JANG_1L, or release readiness. |
| Generation defaults and startup parity contract | `build/current-generation-defaults-contract-after-do-sample-false-mimo-20260607.json` | `status=pass`; CLI/server resolver and MLXStudio startup/session settings both reflect model-owned defaults including `generation_config.do_sample=false` as greedy omitted-request sampling. JANG chat sampling metadata still overrides generic `generation_config`, explicit request/CLI overrides still win, and panel startup must not synthesize hidden `--default` sampler flags from UI/session state. |

## Open objective rows

| Row | Status | Required release evidence |
| --- | --- | --- |
| Cross-family live multi-turn smoke matrix is release-cleared | OPEN | Live server matrix across current release-critical model families with text, multi-turn, tools, tool-result continuation, raw-leak checks, JSON/code/whitespace checks, cache telemetry, and no hidden parser/template rewrites. |
| MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared | OPEN | Long prompt coherence, exact tool/JSON literal values, CB/simple parity, source-vs-quant or equivalent artifact classification, and real VL/audio/video runtime if advertised. |
| MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared | OPEN | Reporter-side parity artifact, installed/public/source hash parity, prompt reproduction or log proof, cancellation cleanup, strict native MiniMax tool dialect parsing, and no raw markup leaks. |
| Real Electron UI unblocked non-MiMo live model matrix is proven | OPEN | LFM MXFP4 dev-Electron proof refreshed on 2026-06-07 and LFM family matrix is now pass. Qwen 3.6 27B and 35B now have deterministic real-UI Responses/tools/image/video/reasoning/cache proofs with native MTP active and `long_tool_loop`, `reasoning_display`, `vl_image`, and `video_where_supported` green. Qwen27 and Qwen35 now also have installed-app image/reasoning/tools/cache and installed-app video passes, with installed-app video enforced in the full checklist. DSV4 missing/memory-gated, largest-context, cancellation cleanup, and installed-app full matrix remain open. |
| Real Electron UI cross-family live model matrix is release-cleared | OPEN | Full installed UI matrix across MiMo, Qwen MTP, Gemma4, Step3.7, LFM, MiniMax, Nemo/Nemotron Omni, DSV4, and other current release families. |
| DSV4 long-output/code/file-generation quality is release-cleared | OPEN | Memory-permitted DSV4 exact code/file/long-output proof with native SWA/CSA/HCA cache, restart L2, tool loops, and UI reflection. |

## Current MiMo V2.5 JANGTQ2 facts

Current local model path:

```text
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2
```

Current MiMo audit:

```text
build/current-mimo-v2-jang2l-current-audit-after-media-runtime-stamp-gate-clean-stale-20260608.json
```

Current green MiMo subproofs:

| Check | Status | Note |
| --- | --- | --- |
| Manifest integrity | GREEN | 115 rows, no missing/mismatched files in current audit. |
| Stale local state cleanup | GREEN | Stale backup/cache targets absent in audit. |
| Structural verify | GREEN | Current structural artifact accepted. |
| SwitchGLU selected expert parity | GREEN | Selected expert parity proof accepted. |
| Text cache narrow proof | GREEN | Narrow text cache proof accepted. |
| Cache-vs-nocache next token | GREEN | Next-token cache parity proof accepted. |
| Tool protocol structure | GREEN | OpenAI tool-call JSON structure parses for failed rows. |
| Prefix/paged/L2 cache reproved | GREEN | In-process prefix/paged/L2 cache stack proof is present, and fresh-process block-disk L2 restore now returns correct visible output with `cache_detail=paged+disk`, `cached_tokens=67`, and disk-backed hit telemetry. |
| Decode speed target | GREEN | Latest decode evidence is above the accepted near-40 tok/s floor. |
| MiMo media capability truth | GREEN | Current text-runtime bundles return runtime modalities `["text"]` and mark preserved image/video/audio as `preserved_unwired`; the capability path no longer imports/registers MiMo media runtime before the metadata gate, so stale HF dynamic-module cache is not recreated by `/capabilities`/audit. |
| API/cache/Responses no-heavy contract | GREEN | Route/telemetry/UI plumbing contract green. |
| Exactness not cache/KV caused | GREEN | Failures reproduce with KV quant, native storage quant, prefix, paged, L2, and hits disabled. |

Current MiMo blockers:

| Blocker | Classification | Required next evidence |
| --- | --- | --- |
| `mimo_long_prompt_coherence_blocked` | model/runtime decode quality or artifact behavior | Long prompt must produce coherent, complete, instruction-following output under release runtime without prompt-only hacks. |
| `mimo_jangtq2_artifact_exactness_blocked` | JANGTQ_2 plain literal copy regression; JANG_2L plain copy passes | Required plain text, tool args, and JSON exact values must preserve literals such as `blue-cat` and `B7-CAT-09`; JSON repair or parser rewriting must not rewrite semantic values to fake a pass. |
| `mimo_jang2l_json_sentinel_exactness_blocked` | JANG_2L tools survive tight memory and JSON no longer first-stops, but exact semantic values remain open | Current fixed-path proof emits parseable JSON and preserves `B7-CAT-09`, but returns `count:1` instead of requested `count:3`. Top-logit trace ranks token `1` above expected token `3` at the count position, so this is no longer the stale non-stream thinking-off forwarding bug and is not a parser, cache, or stop-token issue. It remains model/runtime exact-value quality. Do not clear it with JSON repair or prompt-only folding. |
| `mimo_xml_function_prompt_bloat_reduced` | native MiMo XML schema no longer triggers duplicate fallback injection | Current no-heavy render proof keeps MiMo tool prompt at 307 tokens without fallback text; live `JANGTQ_2` still emits structured tool calls, but literal exactness remains open. |
| `mimo_jang2l_tool_memory_pressure_fixed_partial` | 105G JANG_2L tool rows now survive tight memory | Current proof `build/current-mimo-v25-jang2l-exactness-tight64-parser-cache-skip-variant-probe-20260608/JANGQ_MiMo-V2.5-JANG_2L/result.json` passes both XML tool rows without Metal OOM. This does not release-clear JANG_2L because sentinel JSON still returns the wrong semantic count and broader long-prompt/CB/UI rows remain open. |
| `mimo_jang2l_runtime_model_type_detection_fixed_partial` | batch generator now resolves inner language-model `mimo_v2` type when the outer VLM wrapper config is blank | This prevents MiMo runtime processors from being skipped by wrapper metadata drift. Live proof after the fix shows JANG_2L tool rows still pass with exact literals, but it does not fix the isolated JSON sentinel empty-output row. |
| `mimo_cb_system_prompt_working_set_pressure_blocked` | CB route/resource behavior | Continuous-batching route must handle system/tool prompts without empty stop or working-set collapse. |
| `mimo_source_vs_quant_first_divergence_missing_or_failed` | unresolved runtime-vs-artifact boundary | User currently disallowed source-vs-quant due RAM. Need either reauthorization or an equivalent current-artifact classification that is strong enough to decide runtime fix vs model requant. |
| `mimo_audio_waveform_live_e2e` | current MiMo JANGTQ2 audio waveform E2E clear | Current proof routes raw audio through MiMo audio-code preprocessing and returns visible audio-conditioned output. Keep the fail-loud missing-payload classifier as a regression guard. |
| `mimo_media_runtime_implementation_missing` | current metadata is fail-closed text runtime | Current bundles preserve vision/audio/video sidecars and live object-media rows prove the compatibility media path, but `/capabilities` still reports those modalities as `preserved_unwired` until a media-enabled MiMo artifact stamps `mimo_v2_multimodal_runtime` and passes live media/cache/UI proof. |

MiMo cache/exactness boundary:

- Do not blame prefix cache, paged cache, block-disk L2, runtime KV quant, or cache hits as the primary literal-exactness cause.
- The no-prefix/KV-none proof reproduced the same exactness failures with those mechanisms disabled.
- The no-source exactness classifier records the current evidence as
  `model_generated_literal_mutation_after_valid_parser_structure`, not parser
  mutation.
- Do not clear MiMo with prompt folding, hidden sampling overrides, JSON repair of semantic values, forced tool argument rewrites, disabled cache, or hidden failed rows.
- If current evidence proves artifact corruption or quantization damage, tell Eric the exact requant/reupload contract instead of compensating in runtime.

## Cache and Responses release requirements

No-heavy API/cache/Responses contracts are green, but live release still requires per-family E2E.

Current no-heavy contract:

```text
build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json
```

## CLI and MLXStudio startup parity requirements

Startup parity is a release requirement, not a cosmetic UI row.

| Startup surface | Required behavior |
| --- | --- |
| CLI `vmlx serve` | Startup flags and omitted request defaults must resolve from request overrides, explicit CLI/session values, JANG chat metadata, then `generation_config.json`; no hidden sampler, parser, cache, MTP, max-output, or max-context override may be invented. |
| MLXStudio generated launch | Session create/settings/reset flows must preview and launch the same app-owned flags the server expects, including parser, reasoning, cache, MTP, max output, max context, and model-owned generation defaults. |
| `generation_config.do_sample=false` | Omitted sampling resolves to greedy defaults (`temperature=0.0`, `top_p=1.0`, `top_k=0`) unless JANG chat sampling metadata or explicit request/CLI values override it. |
| JANG chat sampling metadata | Takes precedence over generic `generation_config` and must clear generic `doSample` UI state rather than stacking conflicting defaults. |
| Additional args | User text additional args may not override app-owned startup flags for parser, reasoning, cache, MTP, model name/template, server ports, or generation defaults. |
| Release proof boundary | The no-heavy startup contract proves source/UI plumbing only. Each model family still needs live API/UI output proof showing the launched process actually used the intended parser, reasoning, cache, MTP, and token limits. |

Current green endpoint/plumbing checks in that contract:

| Endpoint or surface | Current no-heavy coverage | Release boundary |
| --- | --- | --- |
| `/v1/chat/completions` | Sampling kwargs, stream cache detail usage, output caps overriding server default. | Must still be proven with visible stream and non-stream output per live family. |
| `/v1/responses` | Sampling kwargs, `previous_response_id` history, JSON schema/text-format preservation, stream cache detail usage, output caps overriding server default. | Must still be proven per live family with tools, tool-result continuation, cancellation, structured output, no hidden-only response, and tail inspection. |
| Responses cancel route | Current-source route contract exists in the broader manifest. | Installed/public package parity still matters; stale public app route absence is not a source-engine fix. |
| Cache stats/reuse endpoints | `cache_stats_reuse_skip_telemetry` and `cache_reuse_endpoints` are green. | Endpoint green is telemetry/plumbing only; it does not prove a model family reuses cache correctly. |
| `/health` native cache block | DSV4 native cache, ZAYA typed CCA, plain attention KV, hybrid SSM partial reuse, TurboQuant KV runtime contract, and no generic TQ on hybrid SSM are covered. | Each architecture still needs live typed-cache proof with family-specific schema and restore behavior. |
| Max output vs max context | Request output caps override server default; prompt/context caps stay separate from output caps. | UI settings, CLI args, and API capabilities must agree for the shipped app. |

Every release-critical family must prove:

| Cache/API item | Required proof |
| --- | --- |
| Chat Completions | Non-stream and stream visible output, tool requests, cancellation, max output cap, max context cap. |
| Responses API | Non-stream and stream, `previous_response_id`, tool calls, tool-result continuation, cancellation, structured output, no empty hidden-only response. |
| Anthropic Messages | Output caps, tools where supported, no parser/template drift. |
| Ollama chat/generate | Output caps, streaming, sentinel behavior, no hidden server default drift. |
| Prefix cache | First miss plus second hit with `cached_tokens` and `cache_detail`. |
| Paged cache | Correct page accounting and no partial-hit corruption. |
| Block-disk L2 | Write, fresh-process restore, hit telemetry, unload/reload or restart proof. |
| TurboQuant KV | Encode/decode/restore only for standard KV families where valid; never substitute generic TQ KV for hybrid SWA/SSM/composite caches. |
| Typed native cache | SWA, SSM, DSV4 composite, MiMo asymmetric SWA, Gemma mixed SWA, and other nonstandard families must use their own schema and restore rules. |
| Largest safe context | Cold miss, warm hit, output tail inspection, cancellation cleanup, memory/resource telemetry. |

Per-family cache reuse is only release-green when the same family has all of:

| Required cache proof | Minimum evidence |
| --- | --- |
| First miss | Request starts cold or with explicit skip, no false positive cache hit. |
| Second hit | Later request reports positive `cached_tokens` and expected `cache_detail`. |
| Native schema | `/health` or capabilities expose the expected native cache family/schema. |
| L2 write | Block disk and any companion state stores positive token counts. |
| L2 restore | Fresh process, unload/reload, or restart request hits disk and returns visible correct output. |
| Media salt | Image/video/audio changes do not reuse stale text-only cache entries. |
| Cancellation cleanup | Aborted Responses/Chat stream does not leave stale scheduler/cache state that corrupts the next request. |
| UI route | MLXStudio settings and gateway route expose the same cache/parser/max-token behavior as CLI/API. |

## Tool, parser, structured output, and leak requirements

Every family must prove:

| Behavior | Required proof |
| --- | --- |
| Required tool call | Correct API-native tool call with valid function name and JSON args. |
| Auto tool call | Model chooses tool only when appropriate and does not raw-dump tool dialect. |
| Tool-result continuation | Final answer uses tool output and stops instead of looping. |
| No-tool request | No fake tool insertion or hidden tool fallback. |
| Loop stop | Repeated mock outputs do not cause infinite command/tool loops. |
| Hidden reasoning leak | Thinking/reasoning stays in the correct channel or remains suppressed when disabled. |
| Raw markup leak | No raw XML, DSML, MiniMax, Qwen, or template tags in visible assistant text unless explicitly requested. |
| JSON/XML/code exactness | Valid parse plus exact semantic values; repair can fix syntax, not fabricate correct values. |
| Whitespace-sensitive output | Exact code and whitespace rows checked. |

## Media/VL/audio/video release requirements

Every advertised media-capable family must prove:

| Media item | Required proof |
| --- | --- |
| OpenAI content parts | Images/audio/video accepted through API content parts. |
| MLXStudio app files | Drag/drop or file picker reaches the same server path as API. |
| Data URLs and safe URLs | Supported where intended, rejected safely where unsupported. |
| Media accounting | Media-expanded prompt accounting prevents unsafe OOM while allowing reasonable inputs on 128GB systems. |
| Media-salted cache | Text cache does not reuse across changed image/video/audio content. |
| Post-media recovery | Text request after media failure or rejection returns visible output without rolling back conversation manually. |
| Media plus tools | Tool requests after media prefill work or fail with a typed, recoverable error. |
| Unwired modalities | If runtime is text-only, metadata and UI must say preserved/unwired, not advertise working media. |

## Per-family release checklist

| Family | Current status | Must still prove or fix |
| --- | --- | --- |
| MiMo V2.5 JANGTQ2 | OPEN | Exact literals, long coherence, CB pressure, artifact/runtime classification, real VL/audio/video, UI parity. |
| Qwen 3.6 27B MTP | PARTIAL/GREEN UI TOOLS+IMAGE+VIDEO+REASONING+MTP | Current deterministic real UI proofs confirm model identity, Responses streaming, built-in tool loop, reasoning display, image answer, video answer, settings overrides, native MTP activation under tool-compatible D1, cache-hit telemetry, hybrid SSM cache, TurboQuant attention KV, block-disk L2, SSM companion L2, server cache controls, and no parser/language leak. Installed-app image/reasoning/tools/cache parity, installed-app video, restart-L2 restore, and 31k-token long-context cache-tail proof are now functionally green and enforced in the full checklist, but cancellation cleanup matrix and packaged acceleration/speed parity are still missing. |
| Qwen 3.6 35B MTP | PARTIAL/GREEN UI TOOLS+IMAGE+VIDEO+REASONING+MTP | Current source, dev-Electron, and installed-app proofs confirm MTP autodetect, `gdn_sink` compatibility, Responses streaming, built-in tool loop, reasoning display, image answer, video answer, deterministic UI sampling, native MTP activation under tool-compatible D1, trained top-k 8, hybrid SSM cache, TurboQuant attention KV, block-disk L2, SSM companion L2, server cache controls, and no parser/language leak. Installed-app image/reasoning/tools/cache parity, installed-app video, and restart-L2 restore are now functionally green and enforced in the full checklist, but 30k-token largest-context tail proof, cancellation cleanup matrix, and packaged acceleration symbol parity are still missing. |
| Nemo/Nemotron Omni | OPEN for media/full matrix | RADIO/vision/audio/video bridge, media cache, tools, structured output, UI. |
| LFM | Needs current full matrix proof | Text/tools/multiturn, JSON/XML/code exactness, cache/L2, API parity, UI. |
| MiniMax/JANGTQ_K | OPEN; CURRENT-SOURCE SMALL TOOL/CACHE/L2 ROW GREEN; #179 ISOLATION MATRIX UPDATED | Current local source/installed app diagnostics and cancel route are clean. Current-source MiniMax Small JANGTQ live smoke `build/current-all-local-model-smoke-minimax-small-jangtq-cache-language-after-bare-invoke-tool-20260609/summary.json` passes with tools, cache repeat, multiturn recall, reasoning, structured JSON, exact code whitespace, `paged+tq` second-hit cache, and `paged+disk+tq` L2 restart after strict complete bare `<invoke>...</invoke>` parser support. Issue179 remains open because reporter-machine parity metadata is missing, reporter installed/public/local server hashes drift, reporter session/log/cancel lifecycle proof is absent, and no concrete prompt currently reproduces the reported wrong-language/numeric screenshot shape locally. Current audit `build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json` now records this source smoke under `current_source_minimax_small_smoke` and in `language_planning_leak_isolation` rows for parser/template/reasoning, paged prefix cache, block-disk L2, and TurboQuant KV. Reporter exact prompt reproduction and reporter generation-config hash/resolved sampling parity remain open. Do not clear from local-only smoke or by disabling cache/L2/TQ/reasoning/sampling without same-prompt single-axis A/B proof. |
| DSV4 Flash | OPEN | Native SWA/CSA/HCA cache, exact code/file/long output, restart L2, tool loops, UI, sufficient memory. |
| Step 3.7 Flash | PARTIAL/GREEN SOURCE VLM ROUTING | Current no-heavy audit `build/current-step37-vlm-runtime-audit-after-source-live-media-proof-20260607.json` is `status=pass` and proves the source/runtime surface is present (`mlx_vlm_step3p7_runtime_available=true`), so stale missing-audit checklist rows are cleared. This is not live media clearance: `live_media_proof.exists=false` and `live_media_proof.pass=false`, and `source_owned_runtime_progress.release_clearance=source_runtime_surface_present_needs_live_proof`. Still must prove text stability, real image/video media path, tools, loop stop, multiturn, API/UI parity, and no raw tool dialect leaks. Do not fake media by metadata override or `has_vision=false`. |
| Gemma 4 12B MXFP4/MXFP8/JANG_4M | PARTIAL/GREEN UI TOOLS+IMAGE+CACHE FOR JANG_4M | Current JANG_4M default-launch Electron proof confirms Responses streaming, built-in tool loop, image answer, server cache controls, settings persistence, `paged+mixed_swa` text/tool cache hits, and block-disk L2 writes. The media turn uses a scoped simple MLLM fallback because optimized batched Gemma4 media prefill produced corrupted visible output; do not claim optimized batched media-cache parity is fixed. MXFP4/MXFP8, audio/video, installed-app parity, and full matrix remain open. |
| Gemma4 E2B/E4B QAT/native MXFP4 | OPEN | Local bundles are present under `/Users/eric/models/JANGQ-AI/gemma-4-E2B-it-qat-MXFP4` and `/Users/eric/models/JANGQ-AI/gemma-4-E4B-it-qat-MXFP4`; registry/proof rows must use Gemma4 parser/reasoning expectations, not stale Gemma3n row IDs. Release proof must be live: text, visual, audio if advertised, tool parser extraction, tool-result continuation, multi-turn recall, JSON/XML/code exactness, cache hit telemetry, block-disk L2 restore, Responses stream/non-stream, content-delta and function-call-argument streaming, generation config defaults, CLI/UI parser/cache/max-token parity, and installed-app startup. |
| Gemma 4 26B/31V VL/video QAT/native MXFP4 | OPEN | Inventory exact bundles and advertised modalities. Prove VL/image and video where advertised, audio only if advertised and coherent, mixed-SWA native cache state, TurboQuant KV encode/decode boundaries, L2 restart restore, tool/reasoning parser behavior, no raw channel/tool leaks, post-media text recovery, media-salted cache, max output/context behavior, and installed-app UI parity. Do not use Gemma4 12B JANG_4M evidence as proof for 26B/31V or QAT/native MXFP4. |
| ZAYA/hybrid/SSM families | OPEN inside cross-family matrix | Typed cache, partial-hit rejection, async rederive or clean prompt-boundary store, media if advertised, UI. |

## Release sequence lock

Only after all objective rows are green:

1. Re-run current release checklist and objective proof.
2. Re-run current regression suite under expected-open policy with no unexpected open rows.
3. Verify current branch/main/push state explicitly.
4. Rebuild packaged Python and MLXStudio app from current source.
5. Verify bundled Python parity.
6. Run installed-app live UI/API/cache/model matrix.
7. Developer ID sign.
8. Notarize and staple.
9. Verify notarized app/DMG.
10. Tag and publish release only after release notes include GitHub `@Hornsan1` credit for reported runtime/media/cache issues.
11. Update public downloads/appcast only after the notarized artifact and release manifest prove green.

## Immediate next blocker choices

Pick one, do not drift:

1. MiMo quality/media blocker: decide runtime vs artifact without source-vs-quant unless Eric reauthorizes RAM, then implement the real fix or give exact requant contract.
2. MiniMax #179 blocker: reproduce reporter parity/root cause and fix runtime/parser/cancel path if it is not stale public/install drift.
3. Real Electron UI matrix: run current installed-app UI/API/cache settings proof for non-MiMo models first, then full cross-family matrix.
4. DSV4 blocker: wait for sufficient memory or use a guarded host, then prove exact code/file/long-output quality with native composite cache.

## Agent control-plane update - 2026-06-07

`AGENTS.md` now contains a hard current-objective execution contract for this
release. Future agents must keep the whole Python engine plus MLXStudio app
release map active, including CLI startup, MLXStudio startup, Chat Completions,
Responses, tools, structured output, cache, TurboQuant KV, media, installed app,
per-family blockers, and signing/notarization lock.

This does not make any model family green by itself. It prevents future work
from drifting into a smaller proof such as source-only tests, one model smoke,
upload chores, or deprecated workspace notes.

Focused validation:

- `.venv/bin/python -m py_compile tests/cross_matrix/run_full_release_objective_checklist.py tests/cross_matrix/release_regression_manifest.py tests/cross_matrix/summarize_objective_proof.py` passed.
- `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py` passed (`2 passed`).
- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py` passed (`3 passed`) and pins AGENTS release-objective surfaces, per-family rows, no-fake-fix rules, and release-lock boundaries.
- `tests/test_agents_release_control_plane.py` is now wired into the current regression suite focused pytest gate and source-hash list.
- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py tests/test_current_regression_suite.py -k "agents_release_control_plane or focused_pytest_gate_sources"` passed (`4 passed`, `73 deselected`).
- The release manifest focused-gate source-hash expectation now explicitly requires `tests/test_agents_release_control_plane.py`.
- `.venv/bin/python -m pytest -q tests/test_release_regression_manifest.py tests/test_current_regression_suite.py -k "focused_pytest_gate_source_hashes or current_suite_source_hash_files"` passed (`1 passed`, `386 deselected`).
- `tests/test_mimo_v2_no_source_exactness_classifier.py` is now wired into the current regression suite focused pytest gate and source-hash list, and the release manifest source-hash expectation.
- `.venv/bin/python -m pytest -q tests/test_mimo_v2_no_source_exactness_classifier.py tests/test_current_regression_suite.py tests/test_release_regression_manifest.py -k "mimo_v2_no_source_exactness_classifier or focused_pytest_gate_source_hashes or focused_pytest_gate_sources"` passed (`4 passed`, `385 deselected`).
- The release manifest MiMo root-cause validator now consumes `build/current-mimo-v2-no-source-exactness-classifier-after-audio-expanded-token-l2-restart-20260608.json`.
- `.venv/bin/python -m pytest -q tests/test_release_regression_manifest.py -k "mimo_v2_root_cause or current_mimo_v2_proof_artifact_constants or current_proof_sweep_tracks_mimo"` passed (`5 passed`, `308 deselected`).
- `.venv/bin/python tests/cross_matrix/run_release_regression_manifest.py --out build/current-release-regression-manifest-after-mimo-audio-expanded-token-l2-restart-20260608.json` regenerated the manifest and correctly remained `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`.
- `.venv/bin/python tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json` regenerated the checklist and correctly remained `status=open`, `failed_count=14`.
- The full checklist now consumes `build/current-mimo-v2-no-source-exactness-classifier-after-audio-expanded-token-l2-restart-20260608.json` in the MiMo group.
- `.venv/bin/python -m pytest -q tests/test_full_release_objective_checklist.py` passed (`2 passed`).
- `.venv/bin/python tests/cross_matrix/run_full_release_objective_checklist.py --out build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json` regenerated the checklist and correctly remained `status=open`, `failed_count=14`.
- `tests/cross_matrix/run_full_release_objective_checklist.py` default output and `tests/cross_matrix/run_current_regression_suite.py` full-checklist command now point at `build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json`.
- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py tests/test_current_regression_suite.py -k "agents_release_control_plane or full_release_objective_checklist"` passed (`5 passed`, `72 deselected`).

## Manifest/checklist sync update - 2026-06-07

The full checklist now consumes `build/current-release-regression-manifest-after-mimo-audio-expanded-token-l2-restart-20260608.json` as its release-manifest input.

Fresh generated checklist:

```text
build/current-full-release-objective-checklist-after-jangtq2-live-release-smoke-20260608.json
```

Result remains `status=open`, `failed_count=14`.

Focused validation:

- `.venv/bin/python -m pytest -q tests/test_agents_release_control_plane.py tests/test_current_regression_suite.py tests/test_full_release_objective_checklist.py tests/test_release_regression_manifest.py -k "agents_release_control_plane or full_release_objective_checklist or mimo_v2_root_cause"` passed (`10 passed`, `382 deselected`).
- Compile validation passed for touched checklist/current-suite/manifest files.

Release boundary is unchanged: no model load, no source-vs-quant, no package build, no signing/notarization/tag/download update.

## Objective proof refresh - 2026-06-07

Refreshed no-heavy contract artifacts in place for current source hashes:

- `build/current-tool-call-contract-after-current-mimo-proof-20260607.json`
- `build/current-max-output-context-contract-after-current-mimo-proof-20260607.json`
- `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`
- `build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json`
- `build/current-cache-architecture-contract-after-mimo-tq-kv-boundary-20260607.json`
- `build/current-model-family-detection-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-parser-registry-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-model-artifact-format-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-generation-defaults-contract-after-do-sample-false-mimo-20260607.json`
- `build/current-native-mtp-contract-after-mimo-capability-snapshot-fix-20260607.json`
- `build/current-vl-media-cache-contract-after-mimo-capability-snapshot-fix-20260607.json`

Fresh objective proof:

```text
build/current-objective-proof-after-mimo-audio-expanded-token-l2-restart-20260608.json
```

Current digest result is 15 PASS / 11 OPEN. The remaining open rows are:

- Cross-family live multi-turn smoke matrix is release-cleared.
- MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared.
- MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared.
- Real Electron UI unblocked non-MiMo live model matrix is proven.
- Real Electron UI cross-family live model matrix is release-cleared.
- DSV4 long-output/code/file-generation quality is release-cleared.

Focused validation passed:

- `.venv/bin/python -m pytest -q tests/test_objective_proof_digest.py tests/test_current_regression_suite.py tests/test_release_regression_manifest.py -k "objective_proof_digest or current_regression_suite_keeps_declared_known_blockers_open or source_hash_list_matches_release_manifest or release_manifest_pointer_matches_current_suite"` -> `108 passed`, `385 deselected`.
- Compile validation passed for objective/current-suite/release-manifest files.

## Startup parity guardrail refresh - 2026-06-07

- Updated local-only `AGENTS.md` to make CLI `vmlx serve` startup and MLXStudio generated startup independent release gates.
- A CLI proof does not clear UI-generated launch/settings/session behavior; a MLXStudio proof does not clear CLI/API startup, request override, `/health`, or capabilities reflection.
- Current no-heavy guardrail test passed: `tests/test_agents_release_control_plane.py` -> `3 passed`.
- Regenerated current manifest/checklist/objective artifacts after the source-hash change; release remains locked with `prepackage_ready=false`, `release_ready=false`, checklist `status=open`, `failed_count=14`, and objective proof 15 PASS / 11 OPEN.

## Generation defaults startup-parity gate refresh - 2026-06-07

- Added `panel_cli_startup_contract` to `tests/cross_matrix/run_generation_defaults_contract.py`.
- The generation-defaults contract now publishes `generation_defaults_family_matrix.cli_mlxstudio_startup_parity` with checks for CLI flag registration, MLXStudio preview/runtime parity, and startup-surface independence.
- Regenerated artifact: `build/current-generation-defaults-contract-after-do-sample-false-mimo-20260607.json`.
- Artifact result: `status=pass`; `panel_generation_defaults` 27 passed, `engine_generation_defaults` 60 passed, `local_generation_metadata_audit` 5 passed, `panel_cli_startup_contract` 9 passed; `missing_markers=[]`.
- Focused validation: `tests/test_generation_defaults_contract.py`, `tests/test_panel_cli_flag_contract.py`, current-suite/release-manifest/objective proof selectors -> `128 passed`, `368 deselected`.
- Regenerated current release manifest, full checklist, and objective proof. Release remains locked: `prepackage_ready=false`, `release_ready=false`, checklist `status=open`, `failed_count=14`, objective proof 15 PASS / 11 OPEN.

## Real-UI unblocked non-MiMo classifier refresh - 2026-06-07

- Fixed `tests/cross_matrix/release_regression_manifest.py` DSV4 real-UI memory preflight validation to accept the current canonical DSV4 path `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K` as well as the legacy HeadBF16 probe suffix.
- DSV4 real-UI memory preflight is now classified as a resource blocker, not a missing unblocked-family proof: status `pass`, launch decision `do_not_launch`, reason `insufficient_memory`.
- Added exact allowed variant handling for Qwen36 real-UI matrix: the family can pass with exactly `Qwen3.6-27B-JANG_4M-MTP` and `Qwen3.6-35B-A3B-MXFP8-MTP`; unexpected or missing variants still remain partial.
- Updated test fixtures for current Qwen36 and LFM25 20260607 proof filenames so synthetic matrix tests cover the same Responses/tools/cache/media surfaces as the current proof rows.
- Regenerated current manifest/checklist/objective artifacts. `Real Electron UI unblocked non-MiMo live model matrix is proven` is now PASS in objective proof.
- Current release state remains blocked: manifest `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; checklist `status=open`, `failed_count=14`.
- Remaining objective OPEN rows: cross-family live multi-turn smoke matrix, MiMo V2.5 runtime/tool/long-prompt quality, MiniMax reporter parity/root cause, Real Electron UI cross-family live model matrix, and DSV4 long-output/code/file-generation quality.
- Focused validation: real-UI matrix/current-suite/objective selectors -> `143 passed`, `350 deselected`; py_compile passed for edited Python files.

## Cross-family live smoke matrix stale ZAYA aggregation refresh - 2026-06-07

- Fixed `tests/cross_matrix/summarize_objective_proof.py` to use current filtered ZAYA text smoke proof `build/current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json` instead of the older combined ZAYA text+VL failure artifact.
- Current objective proof now classifies cross-family live smoke as non-MiMo green and MiMo-only deferred: `non_mimo_status=pass`, `non_mimo_missing_required_family_keys=[]`, `non_mimo_not_pass_artifacts=[]`, `missing_required_family_keys=[mimo_v2]`, `release_boundary=non_mimo_live_smoke_clear_mimo_v2_deferred`.
- This does not clear the cross-family smoke requirement because MiMo V2.5 remains a required family and still fails exact tool/JSON literal rows.
- Regenerated current objective proof, release manifest, and full checklist. Release remains locked: manifest `current_proof_sweep=fail`, `prepackage_ready=false`, `release_ready=false`; checklist `status=open`, `failed_count=14`.
- Focused validation: objective/current-suite/manifest selectors -> `108 passed`, `385 deselected`; py_compile passed for edited objective proof files.

## MiMo local JANG_2L/JANGTQ_2 metadata honesty contract - 2026-06-07

- Patched local `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L/config.json` so it no longer advertises unwired media runtime.
- Backup: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L/config.json.pre-text-runtime-metadata-20260607`.
- Patch proof: `build/current-mimo-v25-jang2l-local-config-text-runtime-patch-20260607.json`.
- Added no-heavy contract `tests/cross_matrix/run_mimo_v2_local_bundle_metadata_contract.py` covering both local MiMo bundles:
  - `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`
  - `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`
- Contract requires runtime modalities `['text']`, preserved/unwired modalities `['vision','audio']`, `multimodal_status='weights_preserved_text_runtime'`, `runtime.multimodal_mode='weights_preserved_text_runtime'`, and preserved `vision_config`, `audio_config`, `preprocessor_config.json`, and `audio_tokenizer/` sidecars.
- Generated proof: `build/current-mimo-v2-local-bundle-metadata-contract-20260607.json`, `status=pass`; `jangtq2` pass; `jang2l` pass.
- Wired the contract into `tests/cross_matrix/run_current_regression_suite.py` as `mimo_v2_local_bundle_metadata_contract`; the step passed inside the current suite.
- Focused validation passed: `tests/test_mimo_v2_local_bundle_metadata_contract.py` plus current-suite/release-manifest source-hash selectors -> `7 passed`, `383 deselected`.
- Full current suite was run and remains `status=open`. New MiMo metadata step passed, but failed steps remain `packaged_integrity_contracts`, `focused_regression_pytest`, `release_regression_manifest`, and `release_gate_skip_app`.
- Release boundary: this is metadata honesty only. MiMo VL/audio/video runtime remains unwired until real `mimo_v2_multimodal.py` or equivalent forward path, media embedding bridge, and media-aware cache/L2 proof exist.

## 2026-06-08 MiMo safe-headroom cache proof refresh

| Item | Artifact / proof | Status |
| --- | --- | --- |
| MiMo mixed-SWA explicit cap regression | `tests/test_mllm_scheduler_cache.py::TestMLLMMixedSWACleanStorePolicy::test_tight_memory_mixed_swa_skips_clean_prompt_above_configured_cap` | PASS after scheduler fix; explicit cap wins over safe-headroom unless force env is set. |
| Focused scheduler proof | `.venv/bin/python -m pytest -q tests/test_mllm_scheduler_cache.py -k 'tight_memory_mixed_swa_skips_clean_prompt_above_configured_cap or mixed_swa_tight_memory_store_uses_clean_prefill_when_headroom_is_safe or tight_memory_mixed_swa_force_env_overrides_cap'` | PASS; 3 passed. |
| Cache architecture contract | `build/current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json` | PASS; `cache_family_pytest` 426 passed, no failed/missing markers. |
| API/cache contract | `build/current-noheavy-api-cache-contract-after-gateway-stale-port-20260609.json` | PASS with current scheduler/test hashes. |
| VL/media cache contract | `build/current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json` | PASS with current scheduler/test hashes. |
| Objective proof digest | `build/current-objective-proof-after-mimo-safe-headroom-contract-refresh-20260608.json` | Cache architecture, generation defaults/Native MTP/VL media, current-source API/cache, and real UI unblocked non-MiMo rows are now PASS. Release remains OPEN on live/model-quality rows. |

Remaining release blockers: cross-family live multi-turn smoke, MiMo V2.5 JANG_2L runtime/tool/long-prompt quality, MiniMax-M2.7-JANGTQ_K reporter parity, real Electron UI full cross-family matrix, and DSV4 long-output/code/file-generation quality. No signing/notarization/tag/download update from this proof.

## 2026-06-09 MiMo/N2 runtime-cache-parser refresh

Scope: active Python/Electron worktree only. No deprecated `/Users/eric/vmlx`, ADLab, Max2, Swift, source-vs-quant, or heavy model launch.

Artifacts refreshed:

- `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json` - status `pass`.
- `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json` - status `pass`.
- `build/current-model-family-detection-contract-after-mimo-n2-runtime-refresh-20260609.json` - status `pass`.
- `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json` - decision `do_not_launch`; indexed payload `118.73GB`, free+speculative memory `94.26GiB`.
- `docs/internal/release-gates/20260608_190041/SUMMARY.md` - skip-app release gate: package/type/bundled import checks pass; objective digest remains fail.

Current classification:

- Built/plumbed: prefix/paged/block-L2 API contracts, hybrid SSM companion cache contracts, generic TurboQuant skip for hybrid SSM, MiMo asymmetric SWA cache status, model-family detection, panel session launch wiring, parser/reasoning CLI startup surfaces, bundled JANGTQ kernels, and MiMo registration.
- N2 JANGTQ2: narrow local proof remains green for text/tool/Responses/cache/small-image and the later clean repeat-image row is pass.
- N2 JANG_1L: not locally proven; blocked by memory preflight, not cleared.
- MiMo JANGTQ_2: runtime/cache/speed path is active but exact literal and tool-argument fidelity fails; not cleared.
- MiMo JANG_2L: not release-cleared; exactness/media/L2 restart rows remain open or missing.

Release boundary: do not sign, notarize, tag, or update downloads from this state. The blocker is no longer missing no-heavy plumbing; it is live model/output/resource/UI proof.

MiMo exactness refinement:

- Existing live proof separates into two blockers.
- No-cache deterministic rows already normalize or mutate literals (`BLUE-CAT` -> `BLUE CAT`, `CERULEAN-472` -> `CERULEAN472`), so this portion is not a detokenizer/parser/cache-only bug.
- The repeated cache row changes from `MIMO-CUTER-17` to `MIMO-CUTR-17` with `cached_tokens=31`, `cache_detail=paged`, and `temperature=0`; this is a live MiMo mixed-SWA paged-cache semantic-fidelity blocker.
- Do not clear MiMo cache from shape/unit tests alone. Next proof needs live no-cache vs paged-hit token trace or a cache-disabled A/B on the same exact prompt to isolate runtime cache from artifact/logit quality.

## 2026-06-09 MiMo lossless auto-cache policy

Change:

- `vmlx_engine/cli.py`: omitted `--kv-cache-quantization` now resolves MiMo V2 asymmetric mixed-SWA to stored-cache `none` instead of q4/q8. Prefix cache, paged cache, and block-disk L2 remain enabled; only lossy storage-boundary quantization is removed from the default path.
- `vmlx_engine/mllm_scheduler.py`: defensive guard disables auto q4/q8 stored-cache quantization for mixed-SWA VLM caches when callers bypass CLI detection.
- `tests/test_turboquant_cache_contract.py`: added MiMo regression while keeping Qwen3.5/N2 hybrid selective live TQ plus stored q4 behavior covered.

Proof:

- Focused pytest passed: MiMo auto lossless row, Qwen3.5 hybrid stored q4 row, and plain Qwen MoE auto-TQ row.
- `build/current-noheavy-api-cache-contract-after-mimo-lossless-auto-kv-20260609.json` - `pass`.
- `build/current-cache-architecture-contract-after-mimo-lossless-auto-kv-20260609.json` - `pass`.
- `build/current-model-family-detection-contract-after-mimo-lossless-auto-kv-20260609.json` - `pass`.
- Bundled Python rebuilt from current source and `panel/npm run verify-bundled` passed.
- `docs/internal/release-gates/20260608_190852/SUMMARY.md` - package/type/bundled import checks pass; objective digest remains fail.

Boundary:

- This fixes the unsafe default policy that allowed lossy q4/q8 stored cache on MiMo mixed-SWA without semantic parity proof.
- This does not yet clear MiMo release. A live repeat-cache exactness run under the new lossless default is still required.
- Explicit MiMo q4/q8 stored-cache quantization remains diagnostic-only until family-specific semantic parity is proven.

## 2026-06-09 MiMo lossless-cache live proof

Artifact:

- `build/current-mimo-v2-jangtq2-cb-cache-lossless-auto-live-20260609.json`

Result:

- Server loaded `MiMo-V2.5-JANGTQ_2` and exited cleanly after the probe.
- Native cache: `mixed_swa_kv_v1`, `cache_subtype=mimo_v2_asymmetric_swa`.
- Prefix cache, paged cache, and block-disk L2 were enabled.
- Stored-cache quantization was disabled: `storage_quantization.enabled=false`, `bits=null`.
- Repeated prompt hit paged cache: `cached_tokens=46`, `cache_detail=paged`.
- Block L2 wrote `78` tokens.
- Repeat output was stable across uncached and cached runs: both returned `ACKCB-742`.
- No hidden `<think>` tags and no 503/working-set rejection.
- Final health generation throughput reported about `58.8 tok/s` for the short run.

Classification:

- The previous MiMo repeat-cache drift under lossy q4/q8 storage is no longer the active default-path blocker under the new lossless policy.
- MiMo is still not release-cleared because literal exactness remains failed before and after cache: expected `ACK-CB-742`, actual `ACKCB-742`.
- Current audit pointer updated to this artifact; refreshed audit is `build/current-mimo-v2-jang2l-current-audit-after-lossless-cache-live-20260609.json`.

### 2026-06-09 MiMo JANGTQ2 token-trace exactness classification
- Proof: `build/current-mimo-v2-jangtq2-cb-cache-lossless-token-trace-live-20260609.json`.
- Server trace: `build/current-mimo-v2-jangtq2-cb-cache-lossless-token-trace-live-20260609.server.log`.
- Status: run passed for process stability, visible output, native cache telemetry, warm cache hit, and L2 write; exact literal row remains failed.
- Cache evidence: `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`, prefix enabled, paged enabled, block-disk L2 enabled, storage quantization disabled, warm cache hit `cached_tokens=46` with `cache_detail=paged`, L2 `78` tokens.
- Token evidence: cold and warm trace generated `ACK`, `CB`, `-`, `7`, `4`, `2`, then stop. Visible output is still `ACKCB-742`; the hyphen before the digits is preserved, while the delimiter between `ACK` and `CB` was not generated.
- Classification: not a prefix/paged/L2/cache-quantization issue and not a parser rewrite. Cache/storage is cleared for this row, but the remaining exactness failure is at token choice / prompt-template / artifact behavior around the second token, not proven text assembly.

### 2026-06-09 MiMo no-source exactness classifier refresh
- Source updated: `tests/cross_matrix/run_mimo_v2_no_source_exactness_classifier.py` now accepts current list-shaped exactness probe artifacts and does not crash when a stale all-local smoke artifact is absent.
- Audit pointer updated: `tests/cross_matrix/run_mimo_v2_jang2l_current_audit.py` now uses `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`.
- Live exactness proof: `build/current-mimo-v25-jangtq2-exactness-variant-probe-live-after-lossless-token-trace-20260609/result.json`.
- Classifier proof: `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`, `status=open`, classification `jangtq2_plain_literal_copy_fails_before_parser_or_json_repair`.
- Current audit proof: `build/current-mimo-v2-jang2l-current-audit-after-lossless-token-trace-classifier-20260609.json`, `status=open`.
- Key failures: completions `blue-cat -> blue cat`, completions/chat `B7-CAT-09 -> B7 CAT-09`, JSON `B7-CAT-09 -> B7CAT-09`, tool args `B7-CAT-09 -> B7CAT-09`.
- Boundary: tool parser/protocol works in this run, but exact argument literals are wrong. This is not JSON repair, not raw XML fallback, not cache quantization, and not prefix/paged/L2 reuse. Source-vs-quant remains skipped by user RAM policy, so the remaining action is either runtime/logit-path root cause with no-source diagnostics or a model rebuild contract if artifact quality is confirmed as the cause.

### 2026-06-09 N2 no-heavy runtime/cache/parser status refresh
- API/cache proof: `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`.
- Cache architecture proof: `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`.
- Family detection proof: `build/current-model-family-detection-contract-after-mimo-n2-runtime-refresh-20260609.json`, `status=pass`.
- N2 JANG_1L memory preflight: `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`, decision `do_not_launch` with `8.0 GiB` Metal/runtime headroom.
- Covered by no-heavy contracts: Chat/Responses sampling kwargs, max-output/max-context separation, JSON schema preservation, streaming cache-detail usage, Responses `previous_response_id`, cache stats/reuse endpoints, TurboQuant KV runtime contract, TurboQuant disk roundtrip, hybrid/native cache matrix, parser registration, CLI parser choices, panel launch policy, and JANG/JANGTQ/MXFP row distinctions.
- JANG_1L live boundary: payload is `118.73GB` decimal / `110.57GiB`; preflight host is `128GiB`; current available memory is `114.64GiB`; required available memory is `118.57GiB` with `8.0GiB` Metal/runtime headroom. A conservative live attempt reached startup and then aborted with Metal OOM, so the row remains release-open until tested with sufficient headroom or a smaller runtime strategy. This is careful-RAM scheduling, not permanent infeasibility.
- 2026-06-09 one-at-a-time retry boundary: user allowed N2 JANG_1L on this 128 GiB host, but current available headroom is still below the post-OOM threshold. `build/current-n2-jang1l-chat-cache-proof-20260609.json` is `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`, `available_gib=111.61`, `required_available_gib=118.57`, `memory_gap_gib=6.96`. The chat/cache gate now enforces this JANG_1L indexed-payload guard before server launch even if generic `--min-available-gb` is low.

### 2026-06-09 N2 explicit autodetect/startup policy row
- Source coverage added: Python model registry, MLXStudio panel detector, and no-heavy family detection contract now include `n2_pro_qwen35_moe_hybrid_vl_policy`.
- Proof: `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`, `status=pass`, `missing_rows=[]`, `n2_pro_qwen35_moe_hybrid_vl_policy=true`.
- Policy pinned: N2 Pro JANG_1L-style metadata resolves as Qwen3.5-MoE hybrid cache with Qwen tool parser, Qwen3 reasoning parser, thinking support, and paged-cache requirement. Because this affine JANG VL-shaped bundle lacks indexed MTP/VL-ready tensors, Python routes it text-only and panel marks `forceTextOnly=true` until real VL support is implemented and live-proven.
- Boundary: this is autodetect/startup/UI-policy coverage only. N2 live runtime, N2 JANG_1L memory-safe load, VL/audio/video, L2 restart restore, installed-app UI proof, and release clearance remain open.

### 2026-06-09 gateway stale-port and standby wake routing contract
- Source fix: `panel/src/main/api-gateway.ts` now treats only active local sessions (`running`, `loading`, `standby`) as gateway-port conflicts. Stopped/error local sessions and remote sessions can keep stale DB ports after restart/sleep without blocking gateway startup.
- Test added: `panel/tests/api-gateway-single-model.behavior.test.ts` -> `allows gateway startup on ports used only by stopped or remote saved sessions`.
- Existing wake path covered: `auto-switches to a standby model by waking it before direct OpenAI streaming`.
- Release gate updated: `tests/cross_matrix/run_noheavy_api_cache_contract.py` now runs `panel_gateway_contracts` and records `gateway_stale_port_startup` plus `gateway_standby_wake_routing`.
- Proof: `build/current-noheavy-api-cache-contract-after-gateway-stale-port-20260609.json`, `status=pass`, `missing_markers=[]`, `gateway_stale_port_startup=true`, `gateway_standby_wake_routing=true`, `panel_gateway_contracts rc=0 passed=2`.
- Boundary: source/panel no-heavy proof only. Installed-app parity, live model routing, model cache/media rows, notarization, and public release remain blocked by the broader objective gates.

## 2026-06-09 current release pointer refresh

- Updated current API/cache release pointers to `build/current-noheavy-api-cache-contract-after-gateway-stale-port-20260609.json`; this is the authoritative no-heavy contract for prefix/cache/responses plus gateway stale-port startup and standby wake routing.
- Updated current model-family release pointers to `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`; this is the authoritative no-heavy contract for MiMo/N2 autodetect, parser/cache metadata, and the explicit N2 text-only-until-VL-proven policy row.
- Historical `docs/internal/release-gates/*/release-ready-manifest.json` snapshots were intentionally left unchanged.
- Release remains blocked until live MiMo/N2 installed-app and media E2E rows are green; this pointer refresh prevents stale no-heavy proof from being treated as current.

## 2026-06-09 MiMo local structural proof refresh

- Reduced blocker: MiMo local manifest/structural proof integrity for the two retained local bundles, without loading the 79G/105G models and without source-vs-quant comparison.
- Updated `tests/cross_matrix/run_mimo_v2_local_bundle_metadata_contract.py` so it now writes:
  - `build/current-mimo-v2-local-bundle-metadata-contract-20260607.json`
  - `build/current-mimo-jangtq2-local-manifest-20260607.tsv`
  - `build/current-mimo-jang2l-local-structural-verify-20260606.json`
- Structural proof now checks both `MiMo-V2.5-JANGTQ_2` and `MiMo-V2.5-JANG_2L` for config/index/sidecar/layout metadata: model-owned `generation_config.json`, `xml_function` tool parser, `think_xml` reasoning parser, hybrid full/SWA cache topology, prefix cache, L2 disk cache, TurboQuant-KV boundary, affine bookend sidecars, stacked `switch_mlp` layout, and absence of legacy `.mlp.experts.*` layout.
- Current audit: `build/current-mimo-v2-jang2l-current-audit-after-mimo-n2-runtime-refresh-20260609.json`, `status=open`, `manifest_integrity=true`, `structural_verify=true`.
- Objective digest remains open. This does not clear live text/tool/exactness/speed/media/L2/UI rows.

## 2026-06-09 no-heavy objective gate refresh

- Regenerated no-heavy contract artifacts after the MiMo structural proof change.
- Newly PASS objective rows:
  - Server default max output and max context are distinct and map to correct CLI flags.
  - Cross-family cache architecture is classified per family.
  - High-risk model family parser, artifact, and launch policy gates are current.
  - Generation defaults, Native MTP, and VL media gates are current.
- Current objective digest remains `build/current-objective-proof-after-mimo-n2-gateway-pointer-refresh-20260609.json` and remains `open` because live-heavy quality/speed/UI/model rows remain open.
- Release lock remains active: no signing, notarization, tagging, or public download update from no-heavy contract pass alone.

## 2026-06-09 LFM25 MXFP4 live smoke refresh

- Ran current bundled-engine live smoke for `LFM2.5-8B-A1B-MXFP4` with tools enabled and media/video disabled.
- Current artifact: `build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json`.
- Passed surfaces: visible cache repeat, multi-turn recall, required tool call, tool-result continuation, structured JSON exactness, parser metadata (`tool_parser=lfm2`, `reasoning_parser=qwen3`), typed `hybrid_ssm_v1` cache, prefix/paged telemetry, `paged+ssm`/`paged+ssm+disk` cache details, and SSM L2 evidence.
- Remaining LFM blockers: exact-code whitespace failed by missing final `)`, block-L2 write/hit checklist remains open, and MXFP8 no-media tools artifact is still missing.
- Fixed a harness false negative: `tool_result_continuation` now expects the exact prompt sentence `STORED blue-cat.`.
- Focused validation: `tests/test_all_local_model_smoke.py` and `tests/test_full_release_objective_checklist.py` passed `75/75`.

## 2026-06-09 LFM25 MXFP8 live smoke refresh

- Ran current bundled-engine live smoke for `LFM2.5-8B-A1B-MXFP8` with tools enabled and media/video disabled.
- Current artifact: `build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json`.
- Passed surfaces: visible cache repeat, multi-turn recall, required tool call, tool-result continuation, structured JSON exactness, parser metadata (`tool_parser=lfm2`, `reasoning_parser=qwen3`), typed `hybrid_ssm_v1` cache, and `paged+ssm` cache telemetry.
- Remaining LFM MXFP8 blocker: exact-code whitespace failed by missing final `)`.
- Full checklist refreshed as `build/current-full-release-objective-checklist-after-lfm25-mxfp8-live-smoke-20260609.json`; it remains open with LFM artifacts current rather than missing.

## 2026-06-09 MiMo/N2 runtime-cache-parser pointer refresh

- Reduced blocker: MiMo/N2 release-gate freshness for prefix/cache/parser/TurboQuant evidence without launching the 79G/105G/118G-class models on an unsafe memory state.
- Regenerated MiMo structural artifacts:
  - `build/current-mimo-v2-local-bundle-metadata-contract-20260607.json`, `status=pass`.
  - `build/current-mimo-jang2l-local-structural-verify-20260606.json`, `status=pass`.
  - `build/current-mimo-jangtq2-local-manifest-20260607.tsv`, verified by the refreshed audit.
- Current MiMo audit is now `build/current-mimo-v2-jang2l-current-audit-after-mimo-n2-runtime-refresh-20260609.json`, `status=open`, with `manifest_integrity=true`, `structural_verify=true`, and `prefix_paged_l2_cache_reproved=true`.
- Current MiMo blockers are still real live rows: text-cache narrow proof, SwitchGLU selected-expert parity, cache-vs-no-cache next-token match, tool protocol/exact arguments, decode speed target, source-vs-quant or accepted no-source equivalent, MLLM inputs-embeds interface proof, block-disk L2 restart restore, image/video E2E, audio waveform E2E, and per-bundle media/L2 rows.
- N2 no-heavy contracts remain green through `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json`, `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json`, and `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`.
- N2 JANG_1L live load remains blocked by memory preflight: `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`, decision `do_not_launch`. Current local free+speculative memory during this refresh was about 50.5 GiB, below the safe threshold for the indexed 118.73 GB payload plus headroom.
- Updated release checklist, objective digest, and release regression manifest pointers to the current MiMo audit/classifier artifacts. This prevents stale proof from being treated as current, but does not clear any live MiMo/N2 release row.

## 2026-06-09 N2 objective/checklist evidence correction

- Reduced blocker: N2 release board accuracy for runtime/cache/parser/UI-startup evidence.
- Root cause: `summarize_objective_proof.py` still hard-coded the N2 row as "no current local artifact or live proof" and pointed the checklist at `build/current-objective-proof-after-mimo-n2-gateway-pointer-refresh-20260609.json`.
- Updated objective digest default to `build/current-objective-proof-after-mimo-n2-runtime-refresh-20260609.json` and wired the N2 row to current evidence:
  - `build/current-n2-pro-jang1l-local-memory-preflight-20260609.json`.
  - `build/current-noheavy-api-cache-contract-after-mimo-n2-runtime-refresh-20260609.json`.
  - `build/current-cache-architecture-contract-after-mimo-n2-runtime-refresh-20260609.json`.
  - `build/current-model-family-detection-contract-after-n2-policy-row-20260609.json`.
- Current N2 truth: local JANG_1L artifact/index is present at `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANG_1L`; indexed payload is 118.73 GB; memory preflight decision is `do_not_launch`; no model load was attempted.
- Current no-heavy truth: API/cache contract, cache architecture contract, model-family detection, N2 family policy, TurboQuant runtime contract, TurboQuant disk roundtrip, and hybrid cache policy are all represented as pass in the N2 row.
- Release boundary: N2 remains open until memory-safe live runtime/API/UI/cache/media proof exists for the relevant JANG_1L/JANGTQ profiles. This is not release clearance.
- Focused verification: objective N2 row test and full checklist N2 row test passed `2/2`; py_compile passed for the edited runners.

## 2026-06-09 N2 JANGTQ2 Responses streaming SSE proof

- Reduced blocker: N2/Qwen-family local source API proof for streaming Responses function-call arguments.
- Source/proof harness: `tests/cross_matrix/run_n2_chat_cache_gate.py` now has `--include-responses-stream-probe`, raw SSE parsing, event-type capture, heartbeat count, argument delta/done/final-item extraction, completed status, and cache telemetry capture.
- Live artifact: `build/current-n2-jangtq2-responses-stream-tool-cache-proof-20260609.json`, `status=pass`.
- Live result: local source server with `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2` passed chat cache/tool, non-stream Responses tool plus `previous_response_id` follow-up, and streaming Responses required-tool SSE.
- Streaming row details: `heartbeat_count=24`; function call `lookup`; parsed args `{"query":"alpha"}`; `response.function_call_arguments.delta`, `response.function_call_arguments.done`, and final `response.output_item.done` all carried non-empty argument text; `cached_tokens=192`; `cache_detail=paged+ssm`.
- Verification: `tests/test_n2_chat_cache_gate.py` passed `8/8`; `py_compile` passed; `git diff --check` passed for the runner/test slice.
- Boundary: N2 remains open. This is local source N2 JANGTQ2 API proof only; it does not clear JANG_1L memory-gated live path, tunnel/gateway SSE parity, installed app/UI execution, L2 restart restore, reasoning/parser leak checks, MTP/`gdn_sink` edge cases, media/VL/audio/video, or release readiness.

## 2026-06-09 Responses server SSE tool proof-map refresh

- Reduced blocker: server-side Responses SSE tool-call proof tracking for `api/ui` and `parser/template` release rows.
- Source/proof-map change: `tests/cross_matrix/run_noheavy_api_cache_contract.py` now runs `responses_streaming_tool_contracts` against server regressions for buffered args, reasoning-channel args, output-index ordering, empty required XML rejection, and preamble + empty XML fail-closed behavior.
- Current proof artifact: `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`, `status=pass`, `missing_markers=[]`, `responses_streaming_tool_call_arguments_and_indexes=true`, `responses_streaming_tool_contracts rc=0 passed=5`.
- Release-manifest expected checks now include `responses_streaming_tool_call_arguments_and_indexes`, so current proof sweeps fail if this row is missing.
- Boundary: this is source/no-heavy proof coverage only. It does not replace raw SSE local-vs-gateway-vs-tunnel capture, installed-app UI execution, N2 JANG_1L memory-safe live proof, or any model-family release clearance row.

## 2026-06-09 N2 JANGTQ2 live proof objective consumption

- Reduced blocker: N2 release-board accuracy. The objective digest, full checklist pointer, current regression suite pointer, and release manifest rows now consume `build/current-objective-proof-after-n2-jangtq2-live-proof-20260609.json` instead of the older PR-intake objective digest.
- The N2 objective row records `build/current-n2-jangtq2-chat-cache-responses-proof-after-responses-parser-20260609.json` with `status=pass`, `stable_text=true`, `tool_probe_pass=true`, `responses_probe_pass=true`, `responses_stream_probe_pass=true`, `cache_hit_cache_detail=paged+ssm`, `cache_hit_cached_tokens=8`, block disk writes/hits, and SSM disk stores.
- Boundary: this is current-source N2 JANGTQ2 chat/cache/Responses proof consumption only. The row remains `OPEN`; JANG_1L runtime/cache/API/UI, media, installed-app/UI, same-model tunnel parity, fresh-process L2 restart, package, signing, notarization, tag, download, and release proof remain required.

## 2026-06-09 N2 JANGTQ2 fresh-process L2 objective consumption

- Live proof: `build/current-n2-jangtq2-chat-cache-responses-l2-proof-20260609.json`, `status=pass`, after one bounded current-source run with `--include-l2-restart-probe` and a `96 GiB` available-memory preflight.
- Restart evidence: `l2_restart_probe_pass=true`; restart row returned visible `ACK`, `cached_tokens=8`, and `cache_detail=paged+ssm+disk`; restart health recorded `block_disk_cache.disk_hits=1` and `ssm_companion_disk.hits=1`.
- Objective proof: `build/current-objective-proof-after-n2-jangtq2-l2-live-proof-20260609.json` records both the existing JANGTQ2 chat/cache/Responses artifact and the new L2 restart artifact while keeping the N2 row `OPEN`.
- Boundary: this is current-source N2 JANGTQ2 only. It does not clear N2 JANG_1L, installed-app/UI, media/VL/audio/video, same-model tunnel parity, package, signing, notarization, tag, download, or release readiness.
- Current N2 objective/checklist also records the skipped JANG_1L live-gate artifact `build/current-n2-jang1l-chat-cache-proof-20260609.json`; N2 remains `OPEN` until memory-safe JANG_1L runtime/cache/API/UI proof exists.

## 2026-06-09 MiMo current-evidence objective cleanup

- Reduced blocker: MiMo release-board accuracy. The objective row no longer lists absent 2026-06-06 diagnostic artifacts as current evidence.
- Current evidence now points at `build/current-mimo-jang2l-local-structural-verify-20260606.json`, `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, `build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json`, and `build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json`.
- Boundary: MiMo remains `OPEN`; current blockers still include JANGTQ2 artifact exactness, decode speed, VL/audio/video wiring and E2E proof, and JANGTQ2/JANG_2L media/L2 rows. Do not clear MiMo from this proof-map cleanup.

## 2026-06-09 Gemma QAT source-smoke objective detail

- Reduced blocker: Gemma QAT/native MXFP4 release-board traceability. The objective row now exposes exact source-smoke artifacts for E2B, E4B, 12B, 26B, and 31B/31V from `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`.
- Media backing is explicitly recorded: E2B/E4B have audio tower and vision weights; 12B has vision weights but only audio embedding metadata; 26B/31B have vision weights and require video runtime proof; full live proof remains red.
- Boundary: Gemma QAT row remains `OPEN`; source-smoke detail is not installed-app/UI/tunnel/full media/cache/Responses/tool proof and is not release clearance.

## 2026-06-09 full checklist refresh after N2/MiMo/Gemma objective details

- Refreshed `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json` from current source.
- Result: `status=open`, `failed_count=122`; the N2 row now references `build/current-objective-proof-after-n2-jangtq2-l2-live-proof-20260609.json`, the MiMo row references `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, and Gemma QAT detail includes current source-smoke/media-backing data.
- Boundary: release remains blocked. This is a no-heavy checklist refresh only, not package/sign/notarize/tag/download work.

## 2026-06-09 Responses preamble empty-XML tool-call boundary

- Reduced blocker: #192/#190 `parser/template` + `api/ui` classification for the new Qwen/Qwen-Coder report where visible preamble text precedes `<tool_call><function=exec_command></function></tool_call>`.
- Verified root-cause boundary instead of trusting the report: current source can parse XML-function syntax into `{}` at the parser layer, but `_filter_to_request_tools()` drops the call when the request schema requires non-empty `cmd`; streaming Responses then emits `tool_calls_required`, not an executable `function_call` with `{}`.
- Added explicit regression `test_streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments` to prove the preamble remains visible, no `response.function_call_arguments.*` events are emitted, no `function_call` item is emitted, and serialized SSE payloads do not include `"arguments": "{}"`.
- Refreshed no-heavy artifact: `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`, `status=pass`, `missing_markers=[]`, `responses_streaming_tool_contracts rc=0 passed=5`.
- Other-agent reminder: do not add a fallback that invents `cmd` from visible preamble text. Missing required tool arguments must fail closed, and any model-side retry behavior is only a retry observation, not a server/parser fix.
- Boundary: this is a source/no-heavy proof-map fix. It does not close #192 from the public-user perspective until the rebuilt installed app is proven, and it does not close #190 live DSV4/default-cache/tool-loop or cross-family release rows.

### 2026-06-10 carry-forward for Qwen3.6 / Qwen coder empty-args reports

- Keep this item active for both 27B and 35B same-family XML tool-call dialects. Do not trust the report's root-cause wording without same-model raw output, but treat the failure shape as release-critical because opencode/Codex harnesses require usable Responses tool streaming.
- Required proof still includes direct server, local gateway, and tunnel raw SSE for the same model and same request with reasoning on: visible content deltas, reasoning deltas where applicable, function-call argument deltas, function-call argument done, final object consistency, valid `output_index` ordering, required/auto/no-tool modes, tool-result continuation, and cache reuse telemetry.
- Server behavior must fail closed when required parameters are absent. Do not synthesize `cmd`, infer arguments from the preamble, disable reasoning to dodge the issue, or strip raw XML after the fact as a fake fix.
- Treat a retry that later emits full XML arguments as a model retry observation only; it is not proof that the first response path is usable.

## 2026-06-09 Responses gateway reasoning empty-final-args boundary

- Reduced blocker: #190/#192 local gateway/panel Responses SSE argument preservation when reasoning events are present and the final function-call item carries `arguments:""` despite prior `response.function_call_arguments.delta/done` events carrying the full JSON.
- Source/proof-map change: added panel regression `passes Responses argument SSE with reasoning and empty final item arguments`, no-heavy check `gateway_responses_reasoning_empty_final_arguments_streaming`, and release-manifest expected check for the same row.
- Proof artifact: `build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json`, `status=pass`, `missing_markers=[]`, `panel_gateway_contracts rc=0 passed=5`, `responses_streaming_tool_contracts rc=0 passed=5`.
- Boundary: this proves current-source local MLXStudio gateway pass-through and panel-side recovery coverage only. It is not a public tunnel proof, not installed-app proof, and not release clearance.

## 2026-06-09 Responses raw SSE parity capture harness

- Reduced blocker: #190/#192 proof classification for the required direct local server vs panel gateway vs tunnel raw SSE comparison.
- Source/proof-map change: added `tests/cross_matrix/run_responses_raw_sse_parity_contract.py` and `tests/test_responses_raw_sse_parity_contract.py`. The classifier reconstructs authoritative tool arguments from `response.function_call_arguments.delta/done` and treats empty final `output_item.done.item.arguments` as acceptable only when done/delta arguments are present.
- Tightened 2026-06-09 follow-up: the default current-suite command now requires expected function `lookup`, expected authoritative arguments `{"query":"alpha"}`, clean SSE JSON parsing, same-model direct/gateway/tunnel captures, valid output item indices, and reasoning-summary SSE events. This specifically guards the no-reasoning-disable workaround boundary and prevents a different tunnel model from satisfying same-model parity.
- Current artifact: `build/current-responses-raw-sse-parity-20260609.json`, `status=open`, `missing_captures=[direct,gateway,tunnel]`, `expected.require_reasoning_events=true`, `expected.require_same_model=true`. This is expected until raw captures are supplied; it must not be treated as pass.
- Mixed-model diagnostic artifact: `build/current-responses-raw-sse-parity-mixed-model-tunnel-output-index-20260609.json`, `status=fail`; direct/gateway Gemma4 E2B captures have valid indices (`message=0`, `function_call=1`), while the public tunnel Qwen35 MXFP8 MTP capture preserves arguments but reports both message and function_call at `output_index=0` and is not same-model.
- Boundary: this is a capture classifier and proof-map pin, not a live tunnel clearance. A passing row still requires all three raw SSE captures for the same model with matching expected function-call arguments, valid output item indices, and reasoning events.

## 2026-06-09 Responses direct SSE Gemma4 tool-argument parser fix

- Reduced blocker: Responses streaming tool-argument loss for Gemma4-family native tool calls.
- Direct local repro before fix: Gemma4 E2B QAT Responses stream with reasoning enabled emitted reasoning/heartbeat events and final visible native tool syntax `<|tool_call>call:record_fact{value:<|"|>blue-cat<|"|>}`, but no `function_call` item and no `response.function_call_arguments.*`; the server correctly failed required tool choice because the parser did not recognize the no-end-marker native call.
- Root cause: `Gemma4ToolParser` required the closing `<tool_call|>` marker. Current Gemma4 QAT can emit a complete closed-brace native call at end-of-output without that marker.
- Source fix: `vmlx_engine/tool_parsers/gemma4_tool_parser.py` accepts complete `<|tool_call>call:name{...}` only at end-of-output when the argument brace is closed. Incomplete/partial calls still fail closed.
- Regression: `tests/test_gemma4_tool_parser.py::TestGemma4ToolParser::test_native_format_complete_call_at_end_without_end_marker`.
- Direct local proof after fix: `build/current-responses-raw-sse-parity-direct-gemma4-e2b-after-parser-20260609.json`, `status=open` only because `gateway` and `tunnel` captures are missing. Direct capture is present, parse-clean, `function_name=record_fact`, `argument_delta_count=2`, `argument_done_count=1`, and authoritative args are `{"value": "blue-cat"}`.
- Boundary: direct local SSE is fixed/proven for this Gemma4 QAT request shape. Gateway and tunnel raw captures remain required; do not close #190/#192 or release from direct proof only.

## 2026-06-09 Responses panel gateway live SSE Gemma4 tool-argument proof

- Reduced blocker: panel gateway raw SSE parity for Gemma4 Responses function-call arguments.
- Method: temporary Vitest-only gateway instantiation with mocked session DB routed to a real current-source vMLX backend serving `/Users/eric/models/JANGQ-AI/gemma-4-E2B-it-qat-MXFP4`; no permanent test harness was added.
- Proof artifact: `build/current-responses-raw-sse-parity-direct-gateway-gemma4-e2b-after-parser-20260609.json`, `status=open` only because `tunnel` capture is missing.
- Direct and gateway captures both have `argument_delta_count=2`, `argument_done_count=1`, parse errors `0`, expected args match, and authoritative args `{"value": "blue-cat"}`.
- Boundary: local direct server and local panel gateway are proven for this request shape. Public tunnel/raw Cloudflare path remains required before closing the parity row or making release claims.

## 2026-06-09 Gemma4 QAT/native MXFP4 E2B/E4B/12B source smoke pass

- Reduced blocker: Gemma4 QAT/native MXFP4 source live rows for E2B, E4B, and 12B.
- Root cause of the last E2B/E4B/12B smoke failure: the tool-result continuation harness asked for `Reply with exactly: STORED blue-cat.` without quoting the target; Gemma4 treated the final dot as sentence punctuation in the tool-result context. Direct A/B showed plain exact-period prompts and quoted tool-result targets preserve the period, while the unquoted tool-result prompt omitted it.
- Source/proof-harness fix: `bench/all_local_model_smoke.py` now quotes the exact target string (`"STORED blue-cat."`) and explicitly says the final period is part of the literal. Validation remains strict; missing-period outputs still fail.
- Regression: `tests/test_all_local_model_smoke.py::test_tool_result_continuation_payload_quotes_exact_target_sentence`.
- Live proof artifacts:
  - `build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`, `status=pass`, `failures=0`.
  - `build/current-all-local-model-smoke-gemma4-e4b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`, `status=pass`, `failures=0`.
  - `build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json`, `status=pass`, `failures=0`.
- Cleared surfaces in these source smokes: text coherency, multi-turn recall, required tool call, tool-result continuation, JSON/code exactness, image/video where emitted, mixed-SWA cache telemetry, and block-disk L2 restart restore. E2B/E4B also clear audio with the `<turn|>` placeholder fix.
- Boundary: this is current-source smoke proof only. Responses direct/gateway/tunnel raw SSE parity, installed-app/UI settings parity, 26B/31B reruns, and package/sign/notarize release gates remain open.

## 2026-06-09 Gemma4 QAT/native MXFP4 26B/31B audio capability and source smoke pass

- Reduced blocker: Gemma4 26B/31B QAT/native MXFP4 old incoherent multilingual/token-soup report and stale audio advertisement.
- Root cause of the remaining 26B/31B smoke failure after loader fixes: `/v1/models/{id}/capabilities` advertised `audio` from `audio_token_id` even when `config.audio_config` was null and the safetensors index had no `audio_tower.*` weights. The smoke then sent audio to models that cannot process audio and correctly got a visible request-for-audio response.
- Source fix: `_bundle_declares_native_audio()` now requires native Gemma4 bundles to have both a real `audio_config` object and `audio_tower.*` weights before advertising audio. Token metadata alone is insufficient. Gemma4 Unified keeps its existing explicit MXFP audio gate behavior.
- Regression: `tests/test_engine_audit.py -k "gemma4_runtime_modalities_do_not_infer_audio_from_token_only_config or gemma4_runtime_modalities_advertise_audio_with_audio_tower_weights or gemma4_runtime_modalities_do_not_infer_video_from_token_only_config"` passed `3/3`.
- Live proof artifacts:
  - `build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json`, `status=pass`, `failed=0`.
  - `build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json`, `status=pass`, `failed=0`.
- Capability proof: both 26B and 31B now report runtime modalities `text`, `vision`, `video`; `media.status_by_modality.audio` is `not_advertised`.
- Cleared surfaces in these source smokes: text coherency, multi-turn recall, reasoning-on visible final, required tool call, tool-result continuation, structured JSON, exact code whitespace, image, video, post-media text recovery, and block-disk L2 restart path.
- Boundary: this is current-source API proof. It does not clear Responses raw SSE parity, installed-app/UI settings parity, package/sign/notarize, or public downloads.

## 2026-06-09 mlx_vlm Gemma4 video processor HF config compat

- Reduced blocker: Gemma4 QAT/native video processor startup compatibility when Hugging Face config supplies extra Gemma4 video processor keys not accepted by the pinned `mlx_vlm` wheel.
- Source fix: `vmlx_engine/runtime_patches/mlx_vlm_compat.py` backports the upstream behavior by filtering unused kwargs for `Gemma4VideoProcessor.__init__`; `vmlx_engine/runtime_patches/__init__.py` installs it with the existing runtime patch bundle.
- Packaging guard: `mlx_vlm_compat.py` is now included in bundled source hash gates for release-gate, bundled Python verification, current regression suite, installed-app parity audit, JANG model compatibility, and packaged integrity.
- Regression: `tests/test_mlx_lm_runtime_patches.py::test_gemma4_video_processor_accepts_hf_config_kwargs`.
- Boundary: this fixes processor construction compatibility only. It is not a substitute for live video semantic proof or installed-app release proof.

## 2026-06-09 mlx_vlm Gemma4 shared-KV mlx-format strict-load compat

- Reduced blocker: Gemma4 QAT/native MXFP4 and mlx-community Gemma4 mlx-format load compatibility.
- Upstream source checked: `Blaizzy/mlx-vlm` PR #1336 (`Fix gemma4 load for mlx-format checkpoints that materialize KV-shared k/v`, opened 2026-06-09).
- Root cause: the existing shared-KV backport filters unused materialized k/v tensors through `sanitize`, but mlx-format checkpoints can skip sanitize in upstream `load_model`, leaving strict `load_weights` to reject `language_model.model.layers.*.self_attn.{k_proj,v_proj,k_norm,v_norm}` tensors for shared-KV layers whose modules are intentionally absent.
- Source fix: `vmlx_engine/runtime_patches/mlx_vlm_compat.py` now reuses one Gemma4 unused shared-KV filter from `Model.sanitize`, `LanguageModel.sanitize`, and `Model.load_weights`. The load-weight path filters only known-unused shared-KV tensors; genuinely missing or unrelated extra weights still raise under strict load.
- Regression: `tests/test_mlx_lm_runtime_patches.py::test_mlx_vlm_gemma4_shared_kv_load_weights_drops_mlx_format_materialized_kv`.
- Verification: full runtime patch file passed `11/11`; current-suite runtime-patch source hash/focused-gate mirror tests passed `3/3`; `py_compile` and `git diff --check` passed for changed files.
- Boundary: no-heavy load compatibility only. This does not clear Gemma live media, installed-app/UI/tunnel, N2, MiMo exactness, DSV4, package, signing, notarization, tag, or download rows.

## 2026-06-09 Single-active cache max_kv_size hybrid guard

- Reduced blocker: cache policy safety for Gemma4, MiMo V2, N2/Qwen3.6, and future mixed sliding/full or hybrid cache families.
- Upstream source checked: `ml-explore/mlx-lm` PR #1343 (`Apply max_kv_size to KVCache layers returned by make_cache()`, opened 2026-06-03).
- Decision: do not blindly backport generic `max_kv_size` rotation into vMLX mixed/hybrid cache paths. The upstream PR notes early-context recall loss when full/global attention layers are converted into bounded rotating windows; that is unacceptable as an implicit release fix for Gemma/MiMo/N2/Qwen cache rows.
- Source fix: `vmlx_engine/utils/single_batch_generator.py` suppresses generic `max_kv_size` for known Gemma4/Gemma4 Unified, MiMo V2, Qwen3.6/N2 model types, explicit `cache_type=hybrid`, hybrid/mixed/SWA/SSM/Mamba subtypes, or mixed `sliding` plus `full/global` layer types. Ordinary KV models still pass the explicit cap through to `mlx_cache.make_prompt_cache`.
- Regressions: `tests/test_single_active_batch_generator.py` covers plain KV pass-through and Gemma4/MiMo/Qwen/future mixed-layer suppression.
- Proof-map fix: current regression-suite source hashes and focused pytest command include `vmlx_engine/utils/single_batch_generator.py` and `tests/test_single_active_batch_generator.py`.
- Verification: focused generator/current-suite guard tests passed `20/20`; `py_compile` and `git diff --check` passed for changed files.
- Boundary: no-heavy cache semantics guard only. This does not clear live N2 JANG_1L, MiMo exactness/media, Gemma installed-app/UI/tunnel, DSV4, package, signing, notarization, tag, or download rows.

## 2026-06-09 Gemma4 Unified packaged parity hash coverage

- Reduced blocker: packaged/bundled runtime parity for `ModuleNotFoundError: No module named 'mlx_vlm.models.gemma4_unified'`.
- Root cause: current source already installs a lazy import hook and resolves `mlx_vlm.models.gemma4_unified` after `import vmlx_engine`; `panel/scripts/verify-bundled-python.sh` also already hash-gates and import-gates the vendored Gemma4 Unified runtime. Installed-app parity and packaged-integrity hash lists were missing the register and vendored runtime files, leaving a silent drift path after packaging.
- Source fix: added `models/gemma4_unified_register.py`, `models/gemma4_unified/__init__.py`, `models/gemma4_unified/config.py`, `models/gemma4_unified/gemma4_unified.py`, and `models/gemma4_unified/processing_gemma4_unified.py` to `panel/scripts/release-gate-python-app.py`, `tests/cross_matrix/run_installed_app_runtime_parity_audit.py`, and `tests/cross_matrix/run_packaged_integrity_contract.py`.
- Regressions: `tests/test_installed_app_runtime_parity_audit.py`, `tests/test_packaged_integrity_contract.py`, and `tests/test_release_gate_python_app.py` now assert those files remain covered.
- Verification: focused test set red before manifest wiring, then passed `4/4` after the fix.
- Boundary: source/package-gate only. This prevents future package omission, but does not rebuild installed app, sign, notarize, tag, publish downloads, or clear Gemma live media/cache/UI/tunnel rows.

## 2026-06-09 Gemma4 QAT/native MXFP4 PLE loader fix

- Reduced blocker: #191 Gemma4 QAT/native MXFP4 local runtime. The first E2B smoke failed during load with affine PLE dequant exhaustion for `language_model.model.per_layer_model_projection.weight` (`weight=(8960,192)`, `scales=(8960,48)`).
- Root cause: the QAT/native artifact stores PLE sidecars as native MXFP4 (`uint32` packed weights, `uint8` UE8M0 scales). Current MLX-VLM Gemma4 instantiates `per_layer_model_projection` and `embed_tokens_per_layer` as quantized modules, so replacing their weights with `float16` dequantized tensors causes runtime `quantized_matmul` failures.
- Source fix: Gemma PLE handling now preserves packed MXFP weights for quantized target modules and configures module `mode=mxfp4`, `bits=4`, `group_size=32`. Older/plain PLE modules still dequantize via native MXFP mode instead of the affine bit-width loop.

## 2026-06-09 Gemma4 MoE cross-shard sidecar and audio waveform source fix

- Reduced blocker: #191/#188 Gemma4 QAT/native MXFP4 loader/media source correctness for 26B-A4B-style MoE bundles and Gemma4 audio inputs.
- Source fix: Gemma4 MoE native-MXFP expert sidecars can now hydrate `.scales`/`.biases` from the safetensors index when the packed `experts.gate_up_proj.weight` or `experts.down_proj.weight` shard is split from its sidecar shard before dequantizing into runtime `experts.switch_glu.{gate,up,down}_proj.weight`.
- Source fix: Gemma4/Gemma4 Unified audio requests now decode temp WAV paths into float32 waveform arrays at the processor sampling rate before calling the native processor. This avoids passing path strings into processors that expect raw waveform arrays.
- Source fix: media prefill guard telemetry now uses a fallback request id for request-like internal test/probe objects that do not expose `request_id`.
- Regressions: `test_gemma4_moe_mxfp_expert_cross_shard_sidecars_are_hydrated`, `test_gemma4_moe_mxfp_vlm_loader_initializes_sidecar_weight_map`, and `test_gemma4_audio_waveforms_from_paths_decodes_wav_to_float32`.
- Proof artifact: `build/current-model-artifact-format-contract-after-gemma4-cross-shard-sidecars-audio-waveform-20260609.json`, `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest rc=0 passed=179 deselected=192`.
- Live 26B QAT proof after sidecar fix: `build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-nomedia-after-cross-shard-expert-sidecars-20260609b/summary.json`, `status=fail` only for two narrow rows. Cleared the prior incoherence class: exact `ACK`, mixed-SWA cache hit `cached_tokens=56` / `cache_detail=paged+mixed_swa`, multi-turn `blue cat`, required tool `record_fact({"value":"blue-cat"})`, JSON exact, code exact whitespace, image `Blue`/`Red`, video fallback `Blue`, and block-disk L2 restart `disk_hits=2`.
- Remaining boundary: tool-result continuation omits final period (`STORED blue-cat` vs `STORED blue-cat.`), and Gemma4 QAT audio still fails honestly because the processor returns no supported audio feature payload. No installed-app behavior or release readiness is claimed.
- Follow-up source change: Gemma4/Gemma4 Unified processor-returned `input_features` and `input_features_mask` are promoted from `extra_kwargs`, salted into media cache keys, and forwarded to the model as native `input_features`/`input_features_mask`; Gemma4 audio prompts missing native audio placeholders append one processor audio token per audio item.
- Follow-up compatibility change: `_run_vision_encoding_inner` reads optional `audio_input_features` fields with `getattr`, preserving older request-like test/probe objects while still forwarding Gemma4 feature tensors.
- Follow-up verification: `tests/test_mllm_scheduler_cache.py -k "audio or processor_direct"` plus explicit Gemma4 `input_features` and placeholder tests passed `9/9`; `tests/test_gemma4_audio_waveform_decode.py` passed `1/1`; `py_compile` and `git diff --check` passed.
- #192 recheck: the quoted preamble plus `<tool_call><function=exec_command></function></tool_call>` assessment stays on the issue list as a #192/#190 subcase, but current source evidence does not support treating it as an executable `arguments:{}` leak. Focused server regressions for output index, required empty XML rejection, and streamed preamble plus empty XML passed `3/3`; current source emits `tool_calls_required` for missing required `cmd`, no `function_call`, no `response.function_call_arguments.*`, and no serialized `"arguments":"{}"` payload for that path.
- Other-agent reminder: do not invent missing `cmd` from visible preamble text. Missing required XML args must fail closed. Public #192 still needs rebuilt/installed-app proof plus raw direct/gateway/tunnel SSE parity before closure.
- No-heavy proof: `build/current-model-artifact-format-contract-after-gemma4-qat-mxfp4-ple-preserve-20260609.json`, `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest rc=0 passed=175 deselected=192`.
- Live E2B proof boundary: `build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-tools-image-after-quantized-ple-preserve-20260609/JANGQ_gemma-4-E2B-it-qat-MXFP4/result.json` loads and serves with the PLE runtime crash gone. The run is still `probe_failed` with one remaining exactness failure: `tool_result_continuation` returned `STORED blue-cat` while the harness expected `STORED blue-cat.`.
- Remaining Gemma QAT/native release rows: E2B exact tool-result punctuation; E4B/12B/26B/31B live runtime rows; full Responses streaming/tool-argument rows; mixed-SWA/paged/L2 cache proof; advertised media or honest modality gates; CLI/UI/installed-app parity. No release/signing/notarization/download action was taken.

## 2026-06-09 native MXFP post-load scale guard

- Reduced blocker: native MXFP4/MXFP8 quantized modules with `uint8` scales could still be reinterpreted by the generic post-load affine bit/group-size heuristic.
- Source fix: `_fix_quantized_bits()` now detects `uint8` MXFP scales on quantized modules, sets `mode=mxfp4` or `mode=mxfp8` from the packed/scales shape relationship, forces `group_size=32`, and removes affine biases before continuing.
- No-heavy proof: `build/current-model-artifact-format-contract-after-native-mxfp-scale-preserve-20260609.json`, `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest rc=0 passed=176 deselected=192`.
- Boundary: this is source/proof-map hardening only. Gemma QAT/native live rows remain open as listed above; no release/signing/notarization/download action was taken.

## 2026-06-09 Gemma4 E4B QAT/native partial live proof

- Ran current-source E4B QAT/native MXFP4 smoke after the loader fixes.
- Artifact: `build/current-all-local-model-smoke-gemma4-e4b-qat-mxfp4-tools-image-after-native-mxfp-scale-preserve-20260609/JANGQ_gemma-4-E4B-it-qat-MXFP4/result.json`, `status=probe_failed`, one failure.
- Positive boundary: E4B loads and serves; no PLE dequant load failure and no runtime `quantized_matmul` failure. Server log records packed Gemma4 PLE preservation for `embed_tokens_per_layer` and `per_layer_model_projection` with `mode=mxfp4`, `bits=4`, `gs=32`.
- Remaining boundary: same exact tool-result punctuation failure as E2B (`STORED blue-cat` vs expected `STORED blue-cat.`). This is not release clearance, and E4B still needs full API/UI/cache/media/installed-app proof.

## 2026-06-09 Gemma4 MoE MXFP expert split and 26B memory boundary

- Reduced source blocker: Gemma4 A4B QAT/native MXFP4 MoE expert tensors can be stored as fused packed `experts.gate_up_proj` and `experts.down_proj`, while the runtime expects `experts.switch_glu.{gate,up,down}_proj.weight`.
- Source fix: loader now dequantizes native-MXFP fused expert tensors and maps them to SwitchGLU float weights before load, instead of leaving packed keys that `strict=False` could ignore.
- No-heavy proof: `build/current-model-artifact-format-contract-after-gemma4-moe-mxfp-expert-split-20260609.json`, `status=pass`, `missing_markers=[]`, `model_artifact_format_pytest rc=0 passed=177 deselected=192`.
- Live 26B boundary: `build/current-all-local-model-smoke-gemma4-26b-a4b-qat-mxfp4-tools-image-after-moe-mxfp-expert-split-20260609/JANGQ_gemma-4-26B-A4B-it-qat-MXFP4/result.json`, `status=probe_failed`, `failures=24`. The model loaded, but the first text prefill terminated with Metal OOM (`Command buffer execution failed: Insufficient Memory`) after a tight-memory allocator drain at about `53.8GB` active baseline.
- Remaining boundary: 26B remains open for memory-safe live proof; this run does not clear text/tool/cache/media/UI/installed-app rows.

## 2026-06-09 Gemma4 12B QAT/native partial live proof

- Ran current-source 12B QAT/native MXFP4 smoke after the loader fixes.
- Artifact: `build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-tools-image-after-native-mxfp-fixes-20260609/JANGQ_gemma-4-12B-it-qat-MXFP4/result.json`, `status=probe_failed`, one failure.
- Positive boundary: 12B loads and serves as `gemma4_unified`; no direct-import startup failure, no PLE dequant crash, and no runtime `quantized_matmul` crash. Text/cache/tool/image probes all reached HTTP 200.
- Remaining boundary: same exact tool-result punctuation failure as E2B/E4B (`STORED blue-cat` vs expected `STORED blue-cat.`). This is not release clearance, and 12B still needs full API/UI/cache/media/installed-app proof.

## 2026-06-09 Gemma4 31B QAT/native partial live proof

- Ran current-source 31B QAT/native MXFP4 smoke after the loader fixes.
- Artifact: `build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-image-after-native-mxfp-fixes-20260609/JANGQ_gemma-4-31B-it-qat-MXFP4/result.json`, `status=probe_failed`, one failure.
- Positive boundary: 31B loads and serves; no loader crash, no Metal OOM, and no runtime `quantized_matmul` crash in the narrow row. Text/cache/tool/image probes all reached HTTP 200.
- Remaining boundary: same exact tool-result punctuation failure as E2B/E4B/12B (`STORED blue-cat` vs expected `STORED blue-cat.`). This is not release clearance, and 31B still needs full API/UI/cache/media/installed-app proof.

## 2026-06-09 Qwen3.6 27B MXFP4 MTP hybrid SSM L2 restart proof

- Source fix: SSM companion disk restart prefix discovery plus SSM L2 budget policy for hybrid VLM models.
- Proof artifact: `build/current-all-local-model-smoke-qwen36-27b-mxfp4-mtp-tools-l2-after-ssm-disk-budget-fix-20260609`.
- Result: `status=pass`, `failures=0`; fresh restart response `ACK`; `prompt_tokens_details.cached_tokens=56`, `cache_detail=paged+ssm+disk`; block disk `disk_hits=1`; SSM disk `hits=1`, `misses=0`; no KV-without-SSM fallback.
- Boundary: current-source Qwen27 MTP smoke/L2 row only. This does not clear installed app/UI parity, Qwen35 deployed parity, MiMo exactness/media, Gemma full matrix, DSV4 memory-gated proof, or release signing/notarization.

## 2026-06-09 Reasoning parser package/hash parity

- Reduced blocker: package/runtime drift for registered reasoning parsers across Qwen3/N2, Gemma4, MiniMax M2, GPT-OSS, Mistral, DeepSeek R1, and think/XML thinking paths.
- Fix: all top-level `vmlx_engine/reasoning/*.py` files are now in bundled-python, release-gate, packaged-integrity, installed-app parity, and current-suite source hash lists.
- Proof: focused guard tests failed before wiring on missing reasoning files, then passed `5/5`; engine audit assertion passed `1/1`; `bash -n`, `py_compile`, and `git diff --check` passed.
- Boundary: parity guard only. It does not prove live Gemma/N2/MiMo behavior, installed-app/UI/tunnel parity, or release readiness.

## 2026-06-09 Gemma4 QAT JANG_4M proof-map lane

- Reduced blocker: Gemma4 QAT JANG_4M inventory/objective/checklist visibility without launching a heavy model.
- Source fix: `tests/cross_matrix/run_gemma_qat_native_mxfp4_inventory_gate.py` now emits `gemma4_12b_qat_jang4m` as a separate release row from native MXFP4 QAT rows.
- No-heavy artifacts refreshed:
  - `build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json`, `status=open`, `missing_required_rows=[]`, `open_required_rows` includes `gemma4_12b_qat_jang4m`.
  - `build/current-objective-proof-after-n2-jang1l-memory-refresh-20260609.json`, `status=open` with the QAT JANG_4M row exposed in Gemma QAT details.
  - `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`, `status=open`, `failed_count=75`.
- Boundary: the existing Gemma4 12B JANG_4M no-media source smoke is still `probe_failed`; this row is not release clearance. Required proof remains autodetect, model-owned `generation_config` defaults, Gemma4 tool/reasoning parser, mixed-SWA/prefix cache, TurboQuant KV boundary where valid, block-disk L2, Responses streaming args/content deltas, media honesty, UI/CLI parity, and installed-app parity.
- Focused validation: `uv run pytest tests/test_gemma_qat_native_mxfp4_inventory_gate.py tests/test_objective_proof_digest.py::test_objective_proof_digest_tracks_gemma_qat_native_mxfp4_release_blocker tests/test_full_release_objective_checklist.py::test_full_release_objective_checklist_blocks_open_gemma_qat_jang4m_row -q` passed `10/10`.

## 2026-06-09 Gemma4 E2B QAT JANG_4M source smoke

- Proof: `build/current-all-local-model-smoke-gemma4-e2b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-E2B-it-qat-JANG_4M/result.json`, `status=pass`.
- Covered source surfaces: Gemma4 autodetect/parser selection, visible text, reasoning separation, required tool call, tool-result continuation, JSON/code exactness, mixed-SWA cache hit telemetry, block-disk writes, and fresh-process L2 restart for E2B QAT JANG_4M.
- Release boundary: source no-media E2B proof only. Gemma4 QAT JANG_4M remains open for media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, larger QAT JANG_4M bundles, and release packaging/signing/notarization.

## 2026-06-09 MiMo media runtime boundary correction

- Source/proof-map fix: MiMo audit now separates source media component presence from runtime media support.
- Current audit: `build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json`, `status=open`.
- Current finding: `source_media_components_present=true`, `raw_audio_request_ingestion=true`, `runtime_capabilities_media_supported=false`, `runtime_media_wired=false`, `media_runtime_implementation=false`, classification `media_components_present_runtime_capabilities_text_only`.
- Side checklist: `build/current-full-release-objective-checklist-after-mimo-media-runtime-boundary-20260609.json`, `status=open`, `failed_count=72`.
- N2 boundary: available memory rechecked at about `111.16 GiB`, below the Nex/N2 Pro 397B JANG_1L `118.57 GiB` launch gate. Do not lower the guard; run one-at-a-time only when headroom is real or a smaller-runtime strategy exists.
- Release boundary: no sign/notarize/package/tag/download action. MiMo still needs live VL/audio/video runtime and L2 proof, exactness artifact/logit/quant or runtime decode fix, decode speed, UI, and installed-app parity.

## 2026-06-09 Gemma4 E4B QAT JANG_4M source smoke

- Proof: `build/current-all-local-model-smoke-gemma4-e4b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-E4B-it-qat-JANG_4M/result.json`, `status=pass`.
- Covered source surfaces: Gemma4 parser/runtime autodetect, visible text, reasoning separation, required tool call, tool-result continuation, JSON/code exactness, mixed-SWA cache hit telemetry, block-disk writes, and fresh-process L2 restart for E4B QAT JANG_4M.
- Release boundary: source no-media E4B proof only. Gemma4 QAT JANG_4M remains open for media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, 12B/26B/31B QAT JANG_4M source smokes, and release packaging/signing/notarization.

## 2026-06-09 Gemma4 12B QAT JANG_4M source smoke blocker

- Proof: `build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-12B-it-qat-JANG_4M/result.json`, `status=probe_failed`.
- Failure: required tool call parsed valid `record_fact({"value":"blue-cat"})`, but visible `<audio|>` leaked in the same assistant turn, failing `tool_visible_text_leak`.
- Boundary: mixed-SWA cache and L2 restart passed in the same run, so do not classify this as cache/L2. Treat as Gemma4 unified parser/template/special-token leak until traced and fixed honestly.

## 2026-06-09 Gemma4 26B QAT JANG_4M source smoke

- Proof: `build/current-all-local-model-smoke-gemma4-26b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-26B-A4B-it-qat-JANG_4M/result.json`, `status=pass`.
- Covered source surfaces: Gemma4 parser/runtime autodetect, visible text, reasoning separation, required tool call, tool-result continuation, JSON/code exactness, mixed-SWA cache hit telemetry, block-disk writes, and fresh-process L2 restart for 26B QAT JANG_4M.
- Release boundary: source no-media 26B proof only. 12B remains blocked by visible `<audio|>` leak, 31B source smoke remains open, and media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, packaging/signing/notarization remain open.

## 2026-06-09 Qwen35 tunnel raw SSE output-index proof

- Proof: `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-20260609.json`, `status=fail`.
- Positive boundary: Qwen35 tunnel raw SSE has reasoning events and complete `record_fact({"value": "blue-cat"})` argument delta/done/final item payloads.
- Failure boundary: the same capture reuses `output_index=0` for both `message` and `function_call`, so output-index validity remains a Responses release blocker.
- Next proof: same-model Qwen35 direct/gateway/tunnel raw SSE after fixing the deployed output-index path. Gemma E2B tunnel wrong-model routing remains a separate Responses blocker.

## 2026-06-09 Gemma4 31B QAT JANG_4M source smoke

- Proof: `build/current-all-local-model-smoke-gemma4-31b-qat-jang4m-tools-nomedia-l2-20260609/JANGQ_gemma-4-31B-it-qat-JANG_4M/result.json`, `status=pass`.
- Covered source surfaces: Gemma4 parser/runtime autodetect, visible text, reasoning separation, required tool call, tool-result continuation, JSON/code exactness, mixed-SWA cache hit telemetry, block-disk writes, and fresh-process L2 restart for 31B QAT JANG_4M.
- Release boundary: source no-media 31B proof only. The remaining QAT JANG_4M source-smoke blocker is 12B visible `<audio|>` leak; media/video, Responses raw SSE args/content deltas, UI/CLI parity, installed-app parity, packaging/signing/notarization remain open.

## 2026-06-09 Gemma4 12B QAT JANG_4M tool sentinel source fix

- Source fix: final response assembly now treats exact singleton Gemma modality sentinels as non-visible channel residue when a valid structured tool call exists.
- Scope: drops only exact `<audio|>`, `<|audio|>`, image, and video sentinel-only content in tool-call responses. It does not hide prose, arbitrary text leaks, or no-tool responses.
- Validation: Gemma4 parser tests `11/11`, focused engine guard tests `2/2`, all-local smoke visible-leak validator `1/1`, and `py_compile` passed.
- Boundary: the parser-fix entry below carries the live 12B rerun and proof-map refresh; this source guard is additional response-assembly defense against parser-selection residue. No release, package, signing, notarization, tag, or download action was run.

## 2026-06-09 Gemma4 modality sentinel tool-leak fix

- Fix: `Gemma4ToolParser._clean_special_tokens` now removes exact Gemma4 modality sentinel residue (`<|image|>`, `<image|>`, `<|audio|>`, `<audio|>`, `<|video|>`, `<video|>`) after native tool-call extraction.
- Regression: `tests/test_gemma4_tool_parser.py::TestGemma4ToolParser::test_native_tool_call_strips_bare_modality_sentinel_leak` failed before the fix and passes after.
- Live proof: 12B QAT JANG_4M rerun at `build/current-all-local-model-smoke-gemma4-12b-qat-jang4m-tools-nomedia-l2-after-modality-token-clean-20260609/JANGQ_gemma-4-12B-it-qat-JANG_4M/result.json`, `status=pass`.
- Current QAT JANG_4M source-smoke status: E2B, E4B, 12B, 26B, and 31B no-media tool/cache/L2 smokes pass; `source_live_smoke_open_rows=[]`. Release remains open for media/video/audio, Responses raw SSE, UI/CLI, installed app, and packaging/signing/notarization.

## 2026-06-09 Responses/Qwen35 raw SSE output-index release gate

- Source/proof-map fix: full release checklist now requires raw Responses SSE output-item index validity and consumes `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-20260609.json` in the Qwen35 group.
- Current proof: Qwen35 tunnel has authoritative `record_fact({"value": "blue-cat"})` args and reasoning events, but reuses `output_index=0` for both message and function_call.
- Regenerated checklist: `build/current-full-release-objective-checklist-after-responses-raw-sse-gemma-surface-20260609.json`, `status=open`, `failed_count=73`, with `qwen35_raw_sse_status_pass` and `qwen35_raw_sse_valid_output_item_indices` red.
- Parallel handoff: `.agents/PARALLEL_RELEASE_LANE_HANDOFF_2026_06_09.md`.
- Boundary: no release, package, signing, notarization, tag, or download action. Next proof must fix/recapture same-model direct/gateway/tunnel raw SSE with valid output indices and no reasoning-disable workaround.

## 2026-06-09 N2 JANG_1L memory refresh after DMG gate

- Refreshed current Nex/N2 Pro 397B JANG_1L no-load preflight without loading weights:
  `build/current-n2-pro-jang1l-local-memory-preflight-after-release-gate-20260609.json`.
- Result: `decision=do_not_launch`, `indexed_payload_gib=110.57`, `required_available_gib=118.57`, `available_gib=112.56`, `memory_gap_gib=6.01`.
- Refreshed the same condition through the chat/cache gate:
  `build/current-n2-jang1l-chat-cache-proof-after-release-gate-20260609.json`,
  `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`,
  `available_gib=112.35`, `memory_gap_gib=6.22`.
- Boundary: no N2 model weights were loaded and no release clearance changed. This remains a careful RAM/headroom scheduling blocker; do not lower the guard or infer failure from total 128 GiB alone.

## 2026-06-09 Qwen35 tunnel raw SSE recapture

- Recaptured public tunnel raw Responses SSE against advertised model
  `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP` with reasoning enabled, required
  `record_fact`, and `max_output_tokens=512`:
  `build/responses-sse-captures-20260609/tunnel-qwen35-mxfp8-mtp-tool-recapture-max512-20260609.sse`.
- Refreshed parity artifact:
  `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-recapture-20260609.json`,
  `status=fail`.
- Finding: tunnel preserves authoritative args `{"value": "blue-cat"}` through
  `response.function_call_arguments.delta`, `.done`, and final function item;
  `reasoning_events=10`, parse errors `0`, model matches expected. The remaining
  failure is duplicate `output_index=0` for both `message` and `function_call`.
- Checklist pointer now consumes the recapture artifact. `qwen35_raw_sse_status_pass`
  and `qwen35_raw_sse_valid_output_item_indices` remain red; local/source guards
  remain green via `build/current-noheavy-api-cache-contract-after-qwen35-output-index-recheck-20260609.json`.
- Boundary: this is not the #192 empty-args failure; it is deployed/tunnel output-index
  freshness plus missing same-model direct/gateway recaptures. No package/sign/release action.
