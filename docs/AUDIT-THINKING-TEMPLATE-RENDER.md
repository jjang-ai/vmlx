# Thinking-Template Render Audit — 2026-05-06

**Spec:** `docs/superpowers/specs/2026-05-06-production-audit-design.md` §8.3, §17.3
**Plan:** `docs/superpowers/plans/2026-05-06-thinking-template-render-audit.md`
**Trigger:** Codex commit `94b16d22` removed the blanket engine-side append of an empty `<think></think>` pair from `vmlx_engine/engine/{batched,simple}.py`. The engine still has a narrow fallback that closes an already-open trailing `<think>` when thinking is explicitly disabled and tools are not present; that fallback is not considered a model-template fix. This audit verifies which bundles honor `enable_thinking=False` at render time and records the source-level normalization needed for templates that use the alias `thinking`.

## Methodology

Two layers:

1. **Layer 1 (tokenizer-only render):** apply chat template with `enable_thinking=True`/`False` via `transformers.AutoTokenizer.apply_chat_template`, using the same normalized kwargs as the engine (`enable_thinking` plus `thinking` alias). For DSV4-Flash, fallthrough to `vmlx_engine.loaders.dsv4_chat_encoder.apply_chat_template` (DSV4 bundles ship `encoding/encoding_dsv4.py` instead of a Jinja template). Assert no unclosed trailing `<think>` for `enable_thinking=False`.
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
| Kimi-K2.6-Small-JANGTQ | kimi_k25 | True | PASS via kwarg alias | template checks `thinking`; engine now mirrors `enable_thinking` to `thinking` |

**Classification key:**
- `omits-think`: template emits no `<think>` for `enable_thinking=False`. Engine layer must ensure model honors via system-role or stop-tokens; live generation verification recommended (Layer 2).
- `emits-empty-pair`: template emits closed `<think></think>` natively. No engine intervention needed.
- `gemma-channels`: template uses Gemma's own `<|channel>thought\n<channel|>` syntax. Engine reasoning extractor (gemma family) handles channel extraction.
- `encoder-driven`: bundle ships a Python encoder (`encoding_dsv4.py`); chat-template path in transformers is N/A.
- `emits-think-on-only`: template emits `<think>` only when `enable_thinking=True`; absent for False.

## Source-level compatibility fix

### Root cause

Some chat templates check the variable `thinking` instead of vMLX's public request field `enable_thinking`. That makes `enable_thinking=False` ineffective unless callers also pass the alias.

The Kimi-K2.6-Small-JANGTQ bundle is one example, but this is a general template-compatibility problem rather than a Kimi-only runtime problem.

### Fix

`vmlx_engine.utils.chat_template_kwargs.build_chat_template_kwargs()` is now the shared construction path for engine tokenizer templates. When the resolved `enable_thinking` value is not `None`, it sets both:

```python
{
    "enable_thinking": resolved_value,
    "thinking": resolved_value,
}
```

The canonical request flag wins over conflicting values supplied inside `chat_template_kwargs`. Reserved engine-owned kwargs (`tokenize`, `add_generation_prompt`) remain protected. Processor/VLM paths that are strict about accepted kwargs continue to receive only `enable_thinking`, not the alias.

Local model-file edits made during investigation are not treated as a shippable vMLX fix. Kimi live/runtime work is deferred per user direction.

## Decision

- **All audited archs with local bundles present honor `enable_thinking=False` at the tokenizer-render layer when rendered with the engine's normalized kwargs.**
- **The blanket empty-pair append removed in `94b16d22` should stay removed.**
- **The remaining close-unclosed-`<think>` fallback in Simple/Batched remains a narrow compatibility fallback and should be audited separately before removal.**
- **`vmlx_engine/model_configs.py` `think_in_template` flag** is unchanged.

## Layer 2 status

Layer 2 (live engine generation) deferred to a follow-up cycle for the three at-risk archs (Ling, Nemotron-Omni, Gemma-4) since Layer 1 confirms template-level correctness. Layer 2 will catch any case where the template is correct but the model auto-thinks anyway (e.g., Ling's `'off'` system-role signal — the model is supposed to read it and skip thinking; live verification confirms the model honors it).

The Layer 2 runs are slow (~2–5 min model load each, ~15–25 min total) and recorded in this same document on next run.

## Remaining items

- Layer 2 live-generation runs for at-risk archs (Ling, Nemotron-Omni, Gemma-4).
- Mistral-Medium-3.5-128B-JANGTQ skipped — `EricsLLMDrive` not mounted.
- Audit and, if possible, replace the remaining close-unclosed-`<think>` fallback with per-template fixes where affected bundles are still active.
