# Model-Family Quirks — Swift Rewrite Audit (2026-04-14)

Non-obvious per-family behaviors the Python engine handles that must be
preserved in the Swift rewrite. Ranked by how many times each quirk has
caused a production regression. Authoritative Python sources live under
`/Users/eric/mlx/vllm-mlx/vmlx_engine/`.

## Global mechanisms

- **Capability detection** — `Sources/vMLXEngine/CapabilityDetector.swift`
  + `ModelCapabilities.swift::ModelTypeTable`. Four tiers: JANG stamp →
  silver allowlist → bronze heuristic → modality refine. Forces
  `cacheType="hybrid"` when Python `is_hybrid_ssm_config` would.
- **§15 reasoning-off fallthrough** — `Sources/vMLXEngine/Stream.swift`
  always routes reasoning deltas to visible content when
  `effectiveThinking == false`. Regressed 3+ times historically.
  See `SWIFT-NO-REGRESSION-CHECKLIST.md` §15.
- **`thinkInTemplate`** — gates the `<think>\n</think>\n\n` pre-fill
  stub. Templates that stamp their own think tags must mark this true
  (Qwen3/3.5, MiniMax, Step, NemotronH).

## Per-family quirks

### MiniMax (M1 / M2 / M2.5)

| Aspect | Python | Swift | Status |
|---|---|---|---|
| reasoning parser | qwen3 | qwen3 | PASS (`ModelTypeTable` entry `minimax*`, priority 20) |
| tool parser | minimax | minimax | PASS |
| thinkInTemplate | true | true | PASS |
| reasoning_effort "none"/"high" via `_ct_kwargs` → template | passed to template as `reasoning_effort` | **NOT plumbed** — Swift never injects `reasoning_effort` into Jinja template kwargs. `reasoningEffort` is parsed from OpenAI routes but only stored on ChatRequest and never consumed | **FAIL** (HIGH severity) |
| enable_thinking → reasoning_effort auto-map | yes (Mistral parser coupling) | no | **FAIL** |
| Flash MoE slot bank for M2.5 (60+ layers × 8 top-k) | Python default 256, plan called for auto-size | **Now auto-sized** from `num_hidden_layers × num_experts_per_tok × 1.5` when user leaves default at 64 | PASS (fixed this session, `EngineFlashMoE.swift`) |

HIGH-severity gap: neither `reasoning_effort` nor request `enable_thinking`
are forwarded into the chat template when rendering. For MiniMax/Mistral4
the template is how the prompt prefix switches between thinking vs
non-thinking modes. Without the plumbing, the model always sees the
default template branch. Swift partially compensates via the `<think>\n`
stub injection for templates that don't stamp their own, but MiniMax
templates DO stamp their own — so reasoning-off uses §15 fallthrough
(works on the wire) but the model still expends tokens on reasoning.

### Qwen 3.5 (text / VL / hybrid SSM)

| Aspect | Python | Swift | Status |
|---|---|---|---|
| reasoning parser | qwen3 | qwen3 (`qwen3_5`, `qwen3_5_moe`, `qwen3_vl`) | PASS |
| tool parser | qwen | qwen | PASS |
| thinkInTemplate | true | true | PASS |
| §15 reasoning-off fallthrough | yes | yes (`Stream.swift:630-633, 1086-1089`) | PASS |
| Hybrid SSM (qwen3_5 VL 24 mamba + 8 KV) | `is_hybrid_ssm_config` detects `layer_types` contains `linear_attention` | `CapabilityDetector.hasLinearAttentionLayer` walks `layer_types` (+ `text_config.layer_types` for VLM) | PASS |
| VL text-only `num_images > 0` guard | yes (Python `batched.py:362`) | Swift VLM processors take text path when `images == []` in `UserInput` | PASS (structural) |
| 122B-A10B is VL model | `is_mllm_model` honors `jang_config.has_vision` | `resolveModelType` walks `jang["has_vision"]` + `architecture.has_vision` | PASS |
| Async SSM re-derive for thinking + hybrid | deferred in Python (too slow hot path) | not implemented | MATCH (equally deferred) |
| think stub double-stamp avoided | `modelStampsThink=true` skips `<think>` injection | same | PASS |

### Nemotron (NemotronH / Cascade 2)

| Aspect | Python | Swift | Status |
|---|---|---|---|
| model_type `nemotron_h` | hybrid, nemotron tool, deepseek_r1 reasoning, thinkInTemplate true | `ModelTypeTable` entry matches exactly | PASS |
| fc1/fc2 JANG rename | `.switch_mlp.up_proj. → .fc1.`, `.down_proj. → .fc2.` | `NemotronH.swift:919-922` remaps identically | PASS |
| SwitchMLP (relu² 2-proj, NOT SwitchGLU) | yes | `NemotronHSwitchMLP` with `isSwitchGLU=false` | PASS |
| 8-bit high-to-low gate dequant | yes (`project_nemotron_jang.md`) | **Not verified in this audit** — needs inspection of JANG loader gate path | PARTIAL — flag for follow-up |
| MTP filter | Python strips MTP weights pre-load | **Not verified** — check JangLoader | PARTIAL |
| 40 Mamba + 8 KV cache reconstruction | BatchKVCache list/tuple fix | hybrid cacheType + CacheCoordinator hybrid branch | PASS (structural) |
| Latent MoE (`fc1_latent_proj`) | yes | `NemotronH.swift:591-658` implements the latent compress → experts → latent expand pattern | PASS |

### Gemma 4

| Aspect | Python | Swift | Status |
|---|---|---|---|
| 128-expert native MoE + SigLIP vision | yes | `Gemma4Text.swift` + `Gemma4.swift` (VLM, 1061 lines) | PASS (ported) |
| `<image>` split bug in mlx_vlm `prepare_inputs` | Python BYPASSES `prepare_inputs`, uses `process_inputs` directly | Swift `Gemma4.swift:1051-1056` reads `<\|image\|>` via `tokenizer.encode` and expands to `imageSeqLength` copies. NO `<image>` substring split anywhere. | PASS (bug absent by construction) |
| reasoning parser `gemma4` | yes | yes (`ModelTypeTable` `gemma4`, `gemma4_text`) | PASS |
| tool parser `gemma4` | yes | yes | PASS |
| Sibling router+experts layout (`FlashMoESwitchGLUShim`) | `_is_gemma4_moe_layer` + shim | `FlashMoE.apply` result includes `gemma4Patched` counter + `FlashMoE` shim exists (engine log confirms) | PASS (structural) |
| Stop tokens | Gemma 4-specific | Not verified here — relies on tokenizer `eos_token_id` list | UNKNOWN (low risk) |
| Flash MoE slot bank | Python recommended bump | auto-sized now | PASS |

### Mistral 4

| Aspect | Python | Swift | Status |
|---|---|---|---|
| MLA attention | `kv_b_proj` split + bfloat16 + fp32 SDPA absorb | `Mistral4.swift:162` `@ModuleInfo(key: "kv_b_proj")` — MLAAttention class present | PASS (structural) |
| `rope_theta` from `rope_parameters` | `TextConfig.__post_init__` pulls out | `Mistral4.swift:118-135` prefers `rope_parameters` then `rope_scaling` then direct `rope_theta` | PASS |
| cacheType `mla` | yes | `ModelTypeTable` `mistral4 cacheType: "mla"` | PASS |
| reasoning parser `mistral` | yes | yes | PASS |
| tool parser `mistral` | yes | yes | PASS |
| reasoning_effort "none"/"high" routed to template | yes (same as MiniMax) | **not plumbed** | **FAIL** (same HIGH as MiniMax) |
| Quant 2L/4M quality poor | known, don't "fix" | — | N/A |
| JANG VLM (`Mistral4VLM.swift`) | yes | file present, 491 lines | PASS (ported) |
| MLLM `is_mllm_model` with JANG has_vision | yes | `CapabilityDetector.resolveModelType` honors `jang.has_vision` | PASS |

## HIGH-severity findings (ranked)

1. **`reasoning_effort` / `enable_thinking` not forwarded to the Jinja
   chat template.** Affects MiniMax and Mistral4 most severely; also
   affects any model whose template branches on these flags (Qwen3.5 is
   partially handled because Swift injects a `<think>\n</think>` stub
   when `thinkInTemplate=false`, and routes stray reasoning via §15 when
   `thinkInTemplate=true`). Fix requires plumbing `additionalContext`
   into `Tokenizer.applyChatTemplate` and every model's chat-template
   call site. OpenAI route already parses `reasoning.effort` — the
   value is just dropped. See `Stream.swift:278` — `effectiveThinking`
   resolves but is never passed to the template renderer.

2. **Nemotron JANG gate dequant (8-bit high-to-low) not verified.**
   Historical regression source. Audit `JangLoader.swift` +
   `NemotronH.swift` initializer for explicit high→low bit ordering
   on the gate tensor.

3. **Nemotron MTP weight filtering.** Python strips MTP layers before
   load (smelt contamination). Swift JangLoader may or may not — not
   verified in this audit.

4. **Flash MoE slot bank default still 64 for explicit user settings.**
   Auto-size now kicks in only when user left the default. If a user
   previously saved `flashMoeSlotBank: 64` in their settings file,
   they will skip auto-sizing. Acceptable but document.

5. **Async SSM re-derive for hybrid + thinking models (Qwen3.5-VL,
   NemotronH with reasoning on).** Matches Python — both are equally
   deferred. No regression vs. Python, but a persistent known gap.

## References

- Python source of truth: `/Users/eric/mlx/vllm-mlx/vmlx_engine/engine/simple.py` (`_ct_kwargs`), `vmlx_engine/model_configs.py` (silver allowlist), `vmlx_engine/utils/ssm_companion_cache.py` (hybrid detection)
- Memory archive: `project_nemotron_jang.md`, `project_mistral4_compat.md`, `feedback_reasoning_off_ui_stuck.md`, `project_cache_matrix_audit_2026_03_28c.md`, `project_flash_moe_slot_bank_sizing.md`, `project_session_2026_04_04.md` (Gemma 4 `<image>` fix)
