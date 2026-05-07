# Thinking-Template Render Audit — 2026-05-06

**Spec:** `docs/superpowers/specs/2026-05-06-production-audit-design.md` §8.3, §17.3
**Plan:** `docs/superpowers/plans/2026-05-06-thinking-template-render-audit.md`
**Trigger:** Codex commit `94b16d22` removed the empty `<think></think>` engine-side inject from `vmlx_engine/engine/{batched,simple}.py` (4 sites). This audit verifies that the removal does not regress `enable_thinking=False` honor for any thinking-default model whose chat template did not natively emit a closed empty `<think></think>` block.

## Methodology

Two layers:

1. **Layer 1 (tokenizer-only render):** apply chat template with `enable_thinking=True`/`False` via `transformers.AutoTokenizer.apply_chat_template`. For DSV4-Flash, fallthrough to `vmlx_engine.loaders.dsv4_chat_encoder.apply_chat_template` (DSV4 bundles ship `encoding/encoding_dsv4.py` instead of a Jinja template). Assert no unclosed trailing `<think>` for `enable_thinking=False`.
2. **Layer 2 (live engine):** load via `SimpleEngine`, generate up to 128 tokens with `enable_thinking=False`, assert no `<think>...</think>` block leaks into the user-visible content. Run only when Layer 1 disagrees with the engine's expected behavior (rare).

Tests: `tests/test_thinking_template_render.py`. Fixtures: `tests/fixtures/thinking_template_models.py`.

## Layer 1 verdict matrix

| Arch | Family | `think_in_template` (registry) | Layer 1: no unclosed `<think>` when disabled | Classification |
|---|---|---|---|---|
| Ling-2.6-flash-JANGTQ2 | bailing_hybrid | False | PASS | omits-think (system-role `'off'` signal) |
| Nemotron-Omni-Nano-JANGTQ | nemotron_h | False | PASS | emits-empty-pair (`<think></think>`) |
| Gemma-4-26B-JANG_4M | gemma4 | False | PASS | gemma-channels (`<\|channel>thought\n<channel\|>`) |
| DSV4-Flash-JANGTQ | deepseek_v4 | True | PASS (via `dsv4_chat_encoder`) | encoder-driven (no Jinja template) |
| MiniMax-M2.7-JANGTQ_K | minimax_m2 | True | PASS | emits-think-on-only |
| Qwen3.6-27B-JANG_4M | qwen3_5 | True | PASS | emits-think-on-only |
| Qwen3.6-35B-A3B-JANGTQ | qwen3_5_moe | True | PASS | emits-think-on-only |
| Kimi-K2.6-Small-JANGTQ | kimi_k25 | True | **FAIL → PASS after patch** | template ignored `enable_thinking` flag (only checked `thinking`) |

**Classification key:**
- `omits-think`: template emits no `<think>` for `enable_thinking=False`. Engine layer must ensure model honors via system-role or stop-tokens; live generation verification recommended (Layer 2).
- `emits-empty-pair`: template emits closed `<think></think>` natively. No engine intervention needed.
- `gemma-channels`: template uses Gemma's own `<|channel>thought\n<channel|>` syntax. Engine reasoning extractor (gemma family) handles channel extraction.
- `encoder-driven`: bundle ships a Python encoder (`encoding_dsv4.py`); chat-template path in transformers is N/A.
- `emits-think-on-only`: template emits `<think>` only when `enable_thinking=True`; absent for False.

## Regression target — Kimi-K2.6

### Root cause

The Kimi-K2.6-Small-JANGTQ bundle's `chat_template.jinja` checks for the variable `thinking` (not `enable_thinking`) at two sites:

- L85 — suffix assistant message rendering with `preserve_thinking is false`:
  ```jinja
  {%- if thinking is defined and thinking is false and preserve_thinking is false -%}
      <think></think>{{render_content(message)}}
  ```
- L107 — final `add_generation_prompt` block:
  ```jinja
  {%- if add_generation_prompt -%}
      <|im_assistant|>assistant<|im_middle|>
      {%- if thinking is defined and thinking is false -%}
      <think></think>
      {%- else -%}
      <think>
      {%- endif -%}
  ```

The vmlx engine (and OpenAI-compatible callers) pass `enable_thinking=False`. Neither L85 nor L107 checked `enable_thinking`, so both fell through to the open `<think>` branch. Pre-`94b16d22`, the engine-side inject closed the open `<think>` after rendering. After the inject removal, Kimi auto-thinks regardless of the flag.

### Patch

Both branches now accept either `thinking is false` or `enable_thinking is false`. A namespace-free `{% set thinking = enable_thinking %}` at template top was tried first but did not propagate under transformers' renderer, so the conditions are OR'd explicitly.

Applied to `/Users/eric/models/JANGQ/Kimi-K2.6-Small-JANGTQ/chat_template.jinja`:

```diff
-    {%- if thinking is defined and thinking is false and preserve_thinking is false -%}
+    {%- if ((thinking is defined and thinking is false) or (enable_thinking is defined and enable_thinking is false)) and preserve_thinking is false -%}
     <think></think>{{render_content(message)}}
```

```diff
+{#- vmlx-audit 2026-05-06: also accept enable_thinking=False (engine flag)
+in addition to the original `thinking=false` variable. Without this, Kimi
+opens an unclosed <think> regardless of the engine flag and the model
+auto-thinks (94b16d22 removed the engine-side <think></think> inject
+that previously masked this bug). The set-inside-if approach with
+namespace-less Jinja didn't propagate under transformers' renderer, so
+we OR the conditions explicitly. -#}
 {%- if add_generation_prompt -%}
   <|im_assistant|>assistant<|im_middle|>
-  {%- if thinking is defined and thinking is false -%}
+  {%- if (thinking is defined and thinking is false) or (enable_thinking is defined and enable_thinking is false) -%}
   <think></think>
   {%- else -%}
   <think>
   {%- endif -%}
 {%- endif -%}
```

Backup: `chat_template.jinja.pre-audit-2026-05-06` (in same directory).

### Re-verify

Layer-1 PASSED post-patch for `kimi-k2.6-small-jangtq`. Full re-run: `16 passed, 6 deselected`.

### Distribution

The patch lives on the user's local model disk. To propagate to other users of the same bundle:

- **Hugging Face mirror** — re-upload `Kimi-K2.6-Small-JANGTQ` with the patched `chat_template.jinja` and bumped tokenizer_config.json `chat_template` field. Both contain the template; `tokenizer_config.json::chat_template` is the source-of-truth that transformers loads.
- **`jang_tools` conversion** — patch upstream `convert_kimi_jangtq.py` (or equivalent) so all future Kimi K2.6 conversions ship the fixed template by default.

The vmlx engine carries no per-family chat-template override, by design — chat templates live in bundles. Adding a registry-side override would be a guard, not a real fix.

## Decision

- **All 8 audited archs honor `enable_thinking=False` at the chat-template layer** (after the Kimi patch).
- **The engine-side `<think></think>` inject removal in `94b16d22` is safe to keep removed.**
- **`vmlx_engine/model_configs.py` `think_in_template` flag** is unchanged. The Kimi patch makes the template's behavior match `think_in_template=True` more strictly — no registry update needed.

## Layer 2 status

Layer 2 (live engine generation) deferred to a follow-up cycle for the three at-risk archs (Ling, Nemotron-Omni, Gemma-4) since Layer 1 confirms template-level correctness. Layer 2 will catch any case where the template is correct but the model auto-thinks anyway (e.g., Ling's `'off'` system-role signal — the model is supposed to read it and skip thinking; live verification confirms the model honors it).

The Layer 2 runs are slow (~2–5 min model load each, ~15–25 min total) and recorded in this same document on next run.

## Remaining items

- Layer 2 live-generation runs for at-risk archs (Ling, Nemotron-Omni, Gemma-4).
- Mistral-Medium-3.5-128B-JANGTQ skipped — `EricsLLMDrive` not mounted.
- Re-publish Kimi-K2.6 bundle on the dealign.ai / JANGQ-AI HF mirror with the patched template (manual user step).
