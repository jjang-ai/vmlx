# API Surface Parity Audit — 2026-05-06

**Spec:** `docs/superpowers/specs/2026-05-06-production-audit-design.md` §9, §17.6, §17.8 (#1, #6).
**Tests:** `tests/test_api_surface_parity.py` (9 tests, all PASS).

## Goal

For semantically equivalent input, the four primary API surfaces must produce
identical OpenAI-shaped messages lists before `apply_chat_template` runs.
Because the chat template is the only prompt-rendering step and cache keys
are derived from the rendered token list, translator equivalence ⇒ prompt
equivalence ⇒ cache-key equivalence ⇒ cache hits across surfaces.

## Surfaces

| Surface | Translator |
|---|---|
| `/v1/chat/completions` | pass-through (canonical messages shape) |
| `/v1/responses` | `vmlx_engine.server._responses_input_to_messages` |
| `/v1/messages` (Anthropic) | `vmlx_engine.api.anthropic_adapter.to_chat_completion` |
| `/api/chat` (Ollama) | `vmlx_engine.api.ollama_adapter.ollama_chat_to_openai` |
| `/api/generate` (Ollama) | `vmlx_engine.api.ollama_adapter.ollama_generate_to_openai` (single-prompt; not in this matrix) |
| `/v1/completions` | single-prompt; not in this matrix |

The single-prompt surfaces (`/api/generate`, `/v1/completions`) have a
different envelope semantics (no messages array) and parity is well-defined
only against the four primary surfaces above.

## Invariants verified

| # | Invariant | Test |
|---|---|---|
| 1 | Plain single-turn user message is identical across all 4 surfaces. | `test_parity_simple_user_message` |
| 2 | System+user is identical: chat-completions inline / Responses `instructions` / Anthropic top-level `system` / Ollama inline all collapse to a leading `{"role": "system"}`. | `test_parity_system_plus_user` |
| 3 | Multi-turn assistant continuation round-trips identically. | `test_parity_multi_turn` |
| 4 | Empty-content assistant edge case does not corrupt the leading user prefix (the part the cache key is derived from). | `test_parity_empty_assistant_history_omitted` |
| 5 | Anthropic `system` accepted as both string and list-of-text-blocks; both produce the same flattened system message. | `test_parity_anthropic_system_as_block_list` |
| 6 | Ollama `options.{num_predict, temperature, top_p, top_k, stop}` map to OpenAI request kwargs; do not affect messages parity. | `test_parity_ollama_options_translated_separately` |
| 7 | `_extract_text_from_content` join convention (`"\n"` separator) is pinned across surfaces. | `test_parity_responses_input_text_join_convention` |
| 8 | Ollama 0.7+ `think: true|false` body field translates to OpenAI `enable_thinking`. | `test_parity_ollama_think_maps_to_enable_thinking` |
| 9 | Anthropic spec default: extended thinking is OPT-IN; translator defaults `enable_thinking=False` when both `thinking` and `enable_thinking` are omitted. | `test_parity_anthropic_thinking_default_off` |

## Findings

### No regressions found

All 9 invariants pass. The four primary translators converge on the same
OpenAI-shaped messages list for equivalent inputs across:

- single-turn text user messages
- system + user (regardless of envelope source)
- multi-turn (system + user + assistant + user)
- ambiguous-content edge cases (empty assistant content, system as block list)
- Ollama option mapping (no leakage into messages)
- thinking-flag normalization (Ollama `think` → `enable_thinking`)
- Anthropic-spec default-off thinking

### Pinned conventions

The audit pinned the following implementation conventions as deliberate and
not to be changed without a cross-surface review:

- **`_extract_text_from_content` join separator** (`"\n"`): when a Responses
  API request carries multiple `input_text` parts in one message, they are
  joined with newline. Real-world clients do not produce this case (they
  combine client-side), but the convention is now under regression coverage.
- **Anthropic default-off thinking** (`enable_thinking=False`): SDK clients
  that omit both `thinking` and `enable_thinking` get a single text block
  with no reasoning content. This default does not bleed across to OpenAI
  `/v1/chat/completions` or Ollama `/api/chat`, where the engine-default
  applies.
- **Ollama `think` body field** maps to OpenAI `enable_thinking`: confirmed
  for `True`, `False`, and absent.

## Gaps not closed by this audit

These are recorded for follow-up cycles, not addressed here:

- **Live cache-hit verification**: the test verifies translator equivalence
  but does not run a real model and assert that turn-2 hits prefix cache from
  a turn-1 call on a different surface. Because the messages are identical,
  the rendered token list will be identical, and the cache key will match —
  but a live regression test would catch any future change that mutates
  prompt rendering downstream of the translator (e.g., a per-surface
  `apply_chat_template` kwarg drift).
- **Tool-call dispatch parity**: Anthropic `tool_use` / OpenAI `tool_calls` /
  Ollama tool dispatch translation is exercised by `tests/test_tool_*.py`
  separately; not duplicated here.
- **Streaming SSE/NDJSON parity**: covered by `tests/test_streaming_*.py`.
- **Non-text content parity (image/audio/video)**: media handling is in
  `tests/test_image_api.py` + `test_vl_video_regression.py`; this audit
  defers to those.

## Test command

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest tests/test_api_surface_parity.py -v
```

Result: `9 passed in ~1.2s`.

## Decision

API surface parity at the translator level is **verified and regression-pinned**.
The cross-surface cache-key derivation invariant in spec §17.8 #1 holds.

No code change required.
