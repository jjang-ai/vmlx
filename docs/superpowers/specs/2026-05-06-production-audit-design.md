# Production Audit Design — vMLX Python Engine + Panel
**Date:** 2026-05-06
**Author:** Eric (Jinho Jang) + Claude (Opus 4.7, 1M context)
**Repo + branch:** `/Users/eric/mlx/vllm-mlx` on `main`
**Base commit:** `94b16d22 Fix Python engine cache and VLM runtime gates`

## 1. Goals

1. Every model architecture's cache contract is **functionally correct in real engine code** — not gated by a "disable feature on this arch" guard, not patched at runtime by a monkey-patch, not silenced by a `try/except: pass`.
2. Every API surface (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/api/chat`, `/api/generate`, `/v1/completions`) yields the same generated content for the same effective input and shares the cache.
3. The Electron panel Chat tab + Server tab probes hold across multi-turn, model switch, reasoning toggle, and tool use.
4. RAM, prefix-hit, paged-hit, L2-disk-hit, and TurboQuant-KV behavior are accounted per (arch, surface, turn) with measurable evidence.
5. Documentation is updated as fixes land. Every commit references an audit-doc entry; every fix has root-cause + regression test.

## 2. Non-goals

- HF model downloads (memory rule — use only what is on disk).
- Signed DMG release / `latest.json` bump (separate task; only on explicit user ask).
- Swift work (`/Users/eric/vmlx/swift` — separate agent).
- Removing or weakening the DSV4-Flash native composite cache routing — that is correct routing for `MLA + SWA + CSA + HSA`, not a guard.

## 3. Working principles

- **No guards.** A branch in `cli.py` that says "disable feature X for arch Y" must be either (a) replaced by real arch-aware feature support, or (b) honestly documented as a deferred port with an explicit follow-up issue. It must never just sit as a permanent feature-disable.
- **No monkey-patches.** `model.make_cache = …`, runtime attribute reassignment in production paths, or `globals()["…"] = …` cross-module signaling is on the chopping block. If a feature needs a hook, the hook lives in the model class or scheduler API, not at import-time runtime mutation.
- **No silenced failures.** `except Exception: pass` is allowed only when the failure is provably non-load-bearing AND there is a log line at INFO or above explaining why the exception was swallowed. Audit will flag the rest.
- **Evidence over assertions.** A claim like "X is fixed" must point to a live model run + cache-stats delta + output transcript. Pytest pass alone is necessary but not sufficient.

## 4. Re-classification of `94b16d22`

| Area | Codex change | Type | Disposition |
|---|---|---|---|
| `cli.py` | DSV4-Flash forces `VMLX_DISABLE_TQ_KV=1` + native composite cache | **Real routing** | Keep. Add cross-API multi-turn coherence test. |
| `cli.py` + `jang_loader.py` | ZAYA fail-fast (`_ensure_zaya_runtime_supported`) | **Honest placeholder** | Replace with real ZAYA Python runtime (CCA + Mamba + ZAYA MoE + EDA + MoD), or pin as deferred port issue. See §8.1. |
| `cli.py` | ZAYA `disable prefix/paged/L2/TQ-KV` block | **Guard** | Replace once ZAYA runtime exists; cache contract extended for ZAYA (KV + conv_state + prev_hs). |
| `cli.py` + `scheduler.py` | Hybrid SSM auto-disable KV-quant + TQ-KV | **Guard** | Replace with hybrid TurboQuant codec covering KV **and** SSM state end-to-end. See §8.2. |
| `engine/batched.py` + `engine/simple.py` | Removed `<think></think>` empty injection (4 sites) | **Behavior change** | **Verify per arch.** If thinking-default models regress on `enable_thinking=False`, restore as a chat-template-driven fix (not the blanket inject). See §8.3. |
| `api/utils.py` + `jang_loader.py` | Qwen3.5/3.6 affine-JANG hybrid → text-only | **Guard** | Replace with real fix to `_load_jang_v2_vlm` divergence vs `_load_jang_v2` for hybrid SSM bundles. See §8.4. |
| `cli.py` | `VMLINUX_DISABLE_TQ_KV` → `VMLX_DISABLE_TQ_KV` rename | **Cleanup** | Keep one release of legacy-name fallback with deprecation log line; drop next release. |
| `server.py` | `_diagnostic_attr` + `_turboquant_kv_cache_status` extraction | **Refactor** | Keep; tests already cover. |
| `server.py` | TQ-KV detection now walks nested MLLM `language_model` | **Real fix** | Keep; tests added. |

The "Real fix" and "Real routing" rows are load-bearing and stay. The "Guard" rows are the targets of this audit.

## 5. Scope

**In scope (architectures, locally available):**
- DSV4-Flash (`/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ`)
- ZAYA-1-8B (`/Users/eric/jang/models/Zyphra/ZAYA1-8B-{base, MXFP4, JANGTQ2, JANGTQ4}`)
- Ling-2.6-flash (`/Users/eric/models/dealign.ai/Ling-2.6-flash-{JANGTQ2, MXFP4}-CRACK`)
- MiniMax-M2.7 + Small + JANGTQ_K (`/Users/eric/models/JANGQ/MiniMax-M2.7-*`)
- Kimi-K2.6-Small (`/Users/eric/models/JANGQ/Kimi-K2.6-Small-JANGTQ`)
- Gemma-4-26B (`/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK`)
- Qwen3.6-27B + 35B-A3B (`/Users/eric/models/dealign.ai/Qwen3.6-{27B-JANG_4M, 27B-MXFP4, 35B-A3B-JANGTQ}-CRACK`)
- Nemotron-Omni-Nano (`/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-{JANGTQ, JANGTQ4, MXFP4}-CRACK`)

**Deferred this session:**
- Mistral-Medium-3.5-128B-JANGTQ — lives on `EricsLLMDrive` which is currently not mounted. Two paths: (a) mount the drive and include, (b) skip from this matrix and tag as separate item. Default = (b) unless user mounts.

**API surfaces under audit:**
- `/v1/chat/completions` (OpenAI)
- `/v1/responses` (OpenAI Responses, with `previous_response_id`)
- `/v1/messages` (Anthropic)
- `/api/chat` (Ollama)
- `/api/generate` (Ollama)
- `/v1/completions` (OpenAI legacy)

**UI surface under audit:**
- Electron panel Chat tab — multi-turn, reasoning toggle, model switch, tool use, file/image attach.
- Electron panel Server tab — external client probes (Copilot, Continue, Claude Code, Anthropic SDK, Ollama).

## 6. Per-arch cache-contract matrix

For each arch, the audit confirms the following cells are **functionally correct** (not guarded around):

| Arch | Family | Cache type | Prefix | Paged | L2 disk | TQ KV | SSM warm pass | Async re-derive (thinking) |
|---|---|---|---|---|---|---|---|---|
| DSV4-Flash | `deepseek_v4` | composite (MLA + SWA + CSA + HSA) | yes (native) | yes (native) | yes (native) | n/a (native owns) | n/a | n/a |
| ZAYA-1-8B | `zaya` | hybrid (CCA + Mamba) | yes (after runtime) | yes (after runtime) | yes (after runtime) | yes (after runtime) | yes (Mamba state) | yes (after runtime) |
| Ling-2.6 | `bailing_hybrid` | hybrid SSM (KV + GLA recurrent) | yes (gpl-strip) | yes (with companion) | yes (with companion) | yes (after hybrid TQ codec) | required | required for thinking models |
| MiniMax-M2.7 | `qwen3_moe` | KV | yes | yes | yes | yes | n/a | n/a |
| Kimi-K2.6 | `kimi_k25` | MLA (`CacheList(KVCache, KVCache)`) | yes | conditional | yes | NO (MLA shape) | n/a | n/a |
| Gemma-4-26B | `gemma4` | rotating_kv (mixed sliding+full) | yes (with rotating metadata) | yes | yes | yes | n/a | n/a |
| Qwen3.6-27B | `qwen3_5` (hybrid) | hybrid SSM | yes | yes | yes | yes (after hybrid TQ codec) | required | required for thinking |
| Qwen3.6-35B-A3B | `qwen3_5_moe` (pure attn MoE) | KV | yes | yes | yes | yes | n/a | n/a |
| Nemotron-Omni-Nano | `nemotron_h` | hybrid (omni text/image/audio/video) | yes | yes | yes | yes (after hybrid TQ codec) | required | required for thinking |

The "after runtime" / "after hybrid TQ codec" / "(native)" annotations are the load-bearing engineering items. They map to §8.

## 7. Per-arch test invariants

Each arch must pass the following before being marked production-ready. Each is a measurable assertion, not a "looks fine".

1. **Load** — clean process load, no errors, peak Metal active memory recorded; peak process RSS recorded.
2. **Single-turn output** — coherent ASCII content (no token salad), reasoning content separate where applicable, `finish_reason=stop` (or family-equivalent), full response readable.
3. **Multi-turn cache hit** — turn-N (N≥2) prefix-hits turn-(N-1) full prompt; `/v1/cache/stats` shows `prefix_hits` increment; `cache_summary.paged.hit_rate` increases monotonically over turns when paged is enabled; SSM companion cache shows `entries > 0` for hybrid models.
4. **Long-output no-loop** — 1k+ token output reads cleanly to end. Validated by unique-word ratio ≥ 0.85 across the last 200 tokens (cheap proxy for "didn't loop"). Validated additionally with full transcript read.
5. **Reasoning on/off** — `enable_thinking=true` produces `reasoning_content` + `content` split correctly; `enable_thinking=false` produces `content` only with no thinking tokens leaking. Both directions verified at:
   - prompt rendering (chat-template output)
   - runtime detokenization (server reasoning extractor)
   - API surface envelope (`reasoning_content` field on chat-completions; `output_text` vs `output_reasoning` on responses).
6. **API surface parity** — same prompt + params via the 4 primary surfaces (chat-completions, responses, messages, ollama-chat) → byte-equal `content` (modulo deterministic SSE framing); same effective cache hit rate.
7. **RAM ceiling** — peak Metal active memory + process RSS under N-turn run ≤ measured baseline + measured per-turn delta. Soak (12 turns × image attachments where applicable) holds Metal flat.
8. **L2 round-trip** — turn-N writes block to disk; new process loads same prompt; cold-restore hit (via `/v1/cache/stats` `disk.hits` increment) within first prefill.

## 8. Real fixes (not guards)

### 8.1 ZAYA Python runtime (replaces fail-fast)

**Status:** No runtime exists in `mlx_lm` or `jang_tools`. The fail-fast in `jang_loader._ensure_zaya_runtime_supported` is honest, but the user directive is real-feature.

**Reference material on disk:**
- `/Users/eric/jang/jang-tools/jang_tools/convert_zaya_{common,jangtq,mxfp4}.py` — converter side, has reference shape info.
- `/Users/eric/jang/jang-tools/examples/zaya/` — example dir.
- ZAYA bundle config: `cca=true, cca_num_q_heads=8, mamba_cache_dtype=float32, num_experts=16, moe_router_topk=1, zaya_use_eda=true, zaya_use_mod=true, zaya_expert_layout=split_switch_mlp, partial_rotary_factor=0.5, num_hidden_layers=80`.

**Implementation outline:**
- `mlx_lm/models/zaya.py` (or `vmlx_engine/runtime_patches/zaya.py` if upstream patch is impractical):
  - `ZayaModelArgs` (dataclass) parsed from `config.json`.
  - `ZayaCCAAttention` — even-numbered layers — KV + 1D conv_state.
  - `ZayaMambaBlock` — odd-numbered layers — Mamba SSM with `prev_hs`.
  - `ZayaMoE` — top-1, 16 experts, `split_switch_mlp` layout, optional EDA + MoD.
  - `ZayaModel` — sequential layer stack, partial-rotary RoPE 0.5, RMSNorm, residual_in_fp32.
  - `make_cache()` returns `[CCACache(KV + conv_state) | MambaCache(prev_hs) for each layer]`.
- Cache contract:
  - `CCACache.update(keys, values, conv_input)` — KV append + 1D conv state shift.
  - `MambaCache.update(prev_hs)` — SSM state replace.
  - Trim/restore methods compatible with prefix + paged + L2 disk.
- Loader side:
  - Remove `_ensure_zaya_runtime_supported` fail-fast.
  - `jang_tools.load_jangtq_zaya` (new) routes JANGTQ ZAYA bundles through native runtime.
  - MXFP4 ZAYA bundles use stock `mlx_lm` weight load + ZAYA model class.
- CLI:
  - Remove ZAYA cache-disable branch.
  - Verify prefix + paged + L2 + TQ-KV all functional.

**Estimated effort:** 1–3 days of port + test work. ~800–1500 LOC.

**If deferred:** The fail-fast remains until next session. The ZAYA fail-fast is documented in this spec as the deferred item.

### 8.2 Hybrid TurboQuant codec (replaces hybrid-SSM auto-disable)

**Status:** `TurboQuantKVCache` compresses positional KV slots only. Hybrid SSM models carry cumulative SSM state alongside KV; live tests on Ling-2.6 JANGTQ2 with auto TQ-KV produced deterministic generated-token loops while the same prompt was coherent with native cache.

**Implementation outline:**
- `vmlx_engine/cache/hybrid_tq_cache.py` (new): `HybridTurboQuantCache` wrapping per-layer `(TurboQuantKVCache, SSMStateCache)` for hybrid models.
- `SSMStateCache` — quantized SSM state with codec parity to TQ-KV (same group_size, same critical-layer protection, same default-bits=3 / critical-bits=4).
- Cache contract integration:
  - `prefix_cache` store/restore: emit + accept both KV and SSM components.
  - `paged_cache` blocks: extend `CacheBlock` with optional SSM payload; ref-count per (KV + SSM).
  - L2 disk store: serialize both components; restore validates round-trip.
- Loader side:
  - `_patch_turboquant_make_cache` extends to hybrid: produce `HybridTurboQuantCache` instead of plain `TurboQuantKVCache`.
- Scheduler side:
  - Remove "hybrid SSM auto-disable" branch.
  - SSM warm pass + async re-derive integrate with the hybrid TQ cache.

**Estimated effort:** 2–4 days. ~600–1000 LOC.

**If deferred:** Hybrid auto-disable remains; we tag as a deferred item; document the live-loop reproducer and the codec design here.

### 8.3 Empty `<think></think>` injection — verify before keep-removed

**Status:** Codex removed 4 sites without explanation. The original injection only fired when `enable_thinking=False AND no tools AND <think> not in prompt`. The risk is that thinking-default models with chat templates that don't auto-emit `<think>` will now ignore `enable_thinking=False`.

**Verification matrix** — fields filled at execution time, per arch with `enable_thinking=False`:

| Arch | Template emits `<think>`? | enable_thinking=false honored without injection? |
|---|---|---|
| DSV4-Flash | (filled by render) | (filled by live test) |
| MiniMax-M2.7 | (filled by render) | (filled by live test) |
| Ling-2.6 | (filled by render) | (filled by live test) |
| Qwen3.6-27B (hybrid) | (filled by render) | (filled by live test) |
| Qwen3.6-35B-A3B | (filled by render) | (filled by live test) |
| Gemma-4-26B | (filled by render — Gemma 4 thinking-capable per family registry) | (filled by live test) |

**Decision policy:**
- If template emits `<think>` and reasoning extractor strips correctly → keep removed.
- If template doesn't emit `<think>` and model thinks anyway despite `enable_thinking=False` → real fix is to either (a) update `chat_template.jinja` for that bundle (preferred), or (b) restore the injection but only for bundles whose templates fail the test (not blanket).

### 8.4 Qwen3.5/3.6 affine-JANG hybrid VLM divergence

**Status:** `_load_jang_v2_vlm` produces corrupt output on affine-JANG hybrid Qwen 3.5/3.6 bundles; `_load_jang_v2` (text path) is coherent with the same weights. JANGTQ/MXTQ Qwen VLM bundles work correctly through the native multimodal path.

**Investigation steps:**
1. Render the same prompt through both loaders' tokenizers — diff token IDs.
2. Diff the model's first-layer activations between the two loaders for the first prompt token (forward to layer-1 hidden state).
3. Compare expert-router quantization, per-projection bit overrides, vision-tower attachment, and embedding tying behavior.
4. Identify the divergence point. Common suspects: bit-pack mismatch on `embed_tokens` between text-loader and VLM-loader; vision-tower's RoPE override stomping text RoPE; `layer_types` field interpretation differing in `_load_jang_v2_vlm`.

**Real fix:** restore byte-equal tensor state across the two loaders for the text-only path. Re-enable affine-JANG VLM image attachment.

**Estimated effort:** 0.5–1 day of bisection + 1–2 days of fix + test.

### 8.5 Async re-derive SSM warm pass for thinking models

**Status:** `_prefill_for_clean_ssm` exists but is unused for `gpl > 0` (thinking-mode generation prompt length > 0). SSM companion cache is skipped because the extracted SSM state is post-generation (contaminated by thinking prefix).

**Real fix paths (pick one based on speed budget):**
- **Capture-during-prefill:** at prefill time, snapshot SSM state at the boundary between user-content and generation-prompt (i.e., before the `<think>`-template tokens). Store both: clean state for cache + post-gen state for current generation.
- **Async re-derive:** while generation runs, in a background task, re-prefill prompt-only and store the clean SSM state for next-turn cache reuse.

**Acceptance criteria:**
- Thinking model + multi-turn → turn-2 SSM companion hit on prompt prefix.
- No throughput regression > 5% on first-turn generation.

## 9. API surface parity invariants

For the same effective input (semantically equivalent — same messages, same system, same params), all four primary surfaces must produce:

1. **Same prompt token sequence** (modulo surface-specific envelope tokens for Anthropic/Ollama).
2. **Same sampling result** with fixed seed + deterministic decoding.
3. **Same `content` text** (byte-equal modulo SSE framing).
4. **Same cache key** for the prompt prefix → cache hit on second surface after first surface populates cache.
5. **Same tool-call structure** when tools are present.
6. **Same reasoning split** (`reasoning_content` + `content`).

**Test harness:** `tests/test_api_surface_parity.py` (new):
- 10 prompts × 4 primary surfaces (chat-completions, responses, messages, ollama-chat) × 3 archs (Ling, MiniMax, DSV4) = 120 samples.
- Secondary surfaces (`/api/generate`, `/v1/completions`) tested separately in arch-specific tests since they have different envelope semantics (single prompt rather than messages array) and parity is well-defined only against the primary 4.
- For each, capture: prompt tokens, sampling output, content, cache_stats delta, reasoning split.
- Assert pairwise equality on items 1–6 with fixed seed and deterministic decoding.

## 10. UI verification plan

**Layered probes** (bottom-up):

1. **Engine direct (Python)** — load model, run multi-turn via `Engine.generate()`, snapshot cache stats. Pinned by `tests/test_live_per_arch_matrix.py` (new).
2. **HTTP server (curl)** — multi-turn + multi-surface; assertions same as §7.
3. **Electron panel Chat tab** — load model in Server tab, multi-turn in Chat tab; verify SQLite session journal matches engine; cache hits via `/v1/cache/stats`.
4. **Electron panel Server tab** — Code Snippets curl probes against the running server.
5. **Mode switching** — model switch mid-chat → cache invalidates cleanly + new chat starts fresh; reasoning toggle round-trips; tool toggle round-trips.

**Regression fixtures:**
- A 6-turn multi-turn template per arch (system + greeting + 5 follow-ups) executed via panel and via curl, with the panel transcript compared to the curl transcript.

## 11. Test infrastructure additions

| New / extended test | Purpose |
|---|---|
| `tests/test_live_per_arch_matrix.py` (new) | Spawn server per arch, run §7 invariants; emit per-arch JSON status + Markdown summary. |
| `tests/test_api_surface_parity.py` (new) | §9 parity assertions across 4 surfaces × 3 archs. |
| `tests/test_zaya_runtime.py` (new, after §8.1) | ZAYA load + cache contract + multi-turn + L2 round-trip. |
| `tests/test_hybrid_tq_codec.py` (new, after §8.2) | Hybrid TurboQuant codec round-trip + Ling/Qwen/Nemotron coherence. |
| `tests/test_thinking_template_render.py` (new, §8.3) | Per-arch chat-template render with `enable_thinking=False` — assert template either emits `<think></think>` or model honors via runtime suppression. |
| `tests/test_qwen_affine_jang_vlm.py` (new, after §8.4) | Qwen 3.5/3.6 affine-JANG VLM image attach + coherence. |
| `tests/test_ssm_async_rederive.py` (new, after §8.5) | Multi-turn thinking model SSM companion hit. |
| `tests/test_engine_audit.py` (extend) | ZAYA real runtime tests once §8.1 lands (replaces fail-fast tests). |
| `tests/test_dsv4_paged_cache.py` (extend) | Add cross-API parity for DSV4. |
| `tests/test_hybrid_prefix_cache.py` (extend) | Add Ling + Qwen 3.6 hybrid + Nemotron Omni. |
| `tests/test_ssm_companion_cache.py` (extend) | Add async re-derive thinking-mode coverage. |
| `tests/test_responses_history.py` (extend) | Add multi-turn parity vs chat-completions. |

## 12. Documentation trail

- **This spec:** `docs/superpowers/specs/2026-05-06-production-audit-design.md` — design, decisions, scoping.
- **Living status board:** `docs/SESSION_2026_05_06_PYTHON_ENGINE_APP_AUDIT.md` — Codex started it; we extend it with per-arch live evidence as runs complete.
- **Per-topic deep-dive sub-docs as items land:**
  - `docs/AUDIT-ZAYA-CCA-RUNTIME.md` (after §8.1)
  - `docs/AUDIT-HYBRID-TQ-CODEC.md` (after §8.2)
  - `docs/AUDIT-THINKING-TEMPLATE-RENDER.md` (§8.3)
  - `docs/AUDIT-QWEN-AFFINE-JANG-VLM.md` (after §8.4)
  - `docs/AUDIT-SSM-ASYNC-REDERIVE.md` (after §8.5)
  - `docs/AUDIT-API-SURFACE-PARITY.md` (after §9)
- **Per-commit policy:** every commit message references the spec section and the audit doc entry.
- **Per-fix policy:** root cause → fix → regression test → audit doc entry, in that order.

## 13. Release-readiness criteria

Before claiming production-ready:

- [ ] All 9 archs (or 8, if Mistral 3.5 deferred) pass per-arch matrix in §7.
- [ ] Hybrid-SSM TurboQuant codec implemented end-to-end (§8.2 lands or is honestly deferred with a tracked issue and the auto-disable is documented).
- [ ] ZAYA runtime implemented (§8.1 lands or is honestly deferred with a tracked issue and the fail-fast is documented).
- [ ] Async-re-derive SSM warm pass works for thinking models with `gpl > 0` (§8.5 lands or deferred).
- [ ] Empty `<think></think>` removal confirmed safe across all thinking archs (§8.3).
- [ ] Qwen3.5/3.6 affine-JANG VLM real fix (§8.4 lands or deferred with explicit text-only-only fallback documented).
- [ ] All 4 primary API surfaces pairwise-equal output on §9 parity harness.
- [ ] Electron panel Chat multi-turn cache hit + reasoning toggle confirmed.
- [ ] No new test failures vs `94b16d22` baseline (854 passed, 48 skipped, 1 pre-existing).
- [ ] Living audit doc has explicit live-source verification per arch + per fix.

## 14. Open scoping decisions for user

These are decisions the user needs to make before execution starts. Each defaults to a value that is safe to proceed under.

1. **ZAYA runtime port (§8.1) — in-session or deferred?**
   - Default: **deferred** (fail-fast remains, tracked as `docs/AUDIT-ZAYA-CCA-RUNTIME.md` with full design).
   - Alternative: **in-session** (1–3 days; replaces fail-fast with real runtime).
2. **Hybrid TurboQuant codec (§8.2) — in-session or deferred?**
   - Default: **deferred** (auto-disable remains, tracked as `docs/AUDIT-HYBRID-TQ-CODEC.md` with full design).
   - Alternative: **in-session** (2–4 days; covers Ling, Qwen 3.6 hybrid, Nemotron Omni hybrid).
3. **Mistral-Medium-3.5** — mount drive or skip from matrix?
   - Default: **skip** (drive not mounted; matrix is 8 archs instead of 9).
   - Alternative: **mount** `EricsLLMDrive` and include.
4. **Empty `<think></think>` injection (§8.3)** — if regression found, restore as blanket inject or per-bundle template fix?
   - Default: **per-bundle template fix** (the right place is `chat_template.jinja`, not engine code).
   - Alternative: **restore blanket inject** (faster but is the original guard pattern).
5. **Order of attack** — depth-first per arch, or breadth-first per fix area?
   - Default: **breadth-first by §8 area** (8.3 verify → 8.4 fix → 8.5 async re-derive → 8.2 hybrid codec if in-session → 8.1 ZAYA if in-session). Each item touches multiple archs.
   - Alternative: **depth-first per arch** (DSV4 fully done → Ling fully done → …). Prone to repeat work since fixes are cross-arch.

## 15. Risks

- **Time budget.** §8.1 + §8.2 alone are 3–7 days of real engineering. If both are in-session, the rest of the audit slips. If both are deferred, we ship "Codex `94b16d22` plus verifiable fixes" and the deferred items remain release blockers for a future cut.
- **Live-test instability.** Some bundles take 5–10 minutes to load on this M5 Max; the matrix has 9 archs × ~12 invariants. We will run them in parallel where possible, sequentially where Metal active memory ceiling forbids.
- **Drive remount.** If `EricsLLMDrive` is mounted mid-session, Mistral 3.5 enters scope and the matrix grows.
- **Memory pressure.** Multi-turn long-output tests (1k+ tokens) plus image attachments can push Metal active memory near the M5 Max ceiling. We watch `mx.metal.get_active_memory()` per turn and fail fast if approaching limits.

## 16. Sequencing (default — assumes ZAYA + hybrid codec deferred)

1. **Setup:** confirm scope decisions (§14); write per-topic deep-dive doc stubs.
2. **§8.3 (verify):** per-arch chat-template render with `enable_thinking=False`. Record matrix in `docs/AUDIT-THINKING-TEMPLATE-RENDER.md`. Decide restore-injection or template-fix.
3. **§8.4 (fix):** Qwen3.5/3.6 affine-JANG VLM divergence root-cause + fix.
4. **§8.5 (fix):** SSM async re-derive for thinking models.
5. **§9 (test):** API surface parity harness, 4 surfaces × 3 archs.
6. **§7 (test):** per-arch matrix run for all 8 archs (excluding Mistral 3.5 by default).
7. **§10 (verify):** Electron panel Chat + Server tab probes.
8. **Release-readiness review** against §13.
9. **Sign-off and (separately) build the signed DMG** if user approves a release cut.

---

**Approval gate:** before execution starts, user reviews this spec and answers the 5 questions in §14.
