# DeepSeek-V4-Flash-0731 — Encoder Contract and Prefix-Cache Invariants

Source of truth: the bundle's own `encoding/README.md` and `encoding/encoding_dsv4.py`
(mirrored at <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/README.md>).
For local reproduction, point `DSV4_BUNDLE` at a verified bundle copy; no
machine-local source path is part of this public contract.

vMLX adapter: `vmlx_engine/loaders/dsv4_chat_encoder.py`.
Tool parser: `vmlx_engine/tool_parsers/dsml_tool_parser.py` (parser key `dsml`,
declared by `jang_config.chat.tool_calling.parser`).

This document exists because several DSV4 runtime defects were *encoder-shaped*:
they were invisible in the engine, the scheduler, and the cache layer, and only
became obvious once the rendered prompt was diffed turn over turn.

---

## 1. Special tokens

| Token | Purpose |
|-------|---------|
| `<｜begin▁of▁sentence｜>` | BOS, prepended once at the very front |
| `<｜end▁of▁sentence｜>` | EOS, ends an assistant turn |
| `<｜User｜>` | User turn prefix |
| `<｜Assistant｜>` | Assistant turn prefix |
| `<｜latest_reminder｜>` | Latest reminder (date, locale, …) |
| `<think>` / `</think>` | Reasoning block delimiters |
| `｜DSML｜` | DSML markup token |

`vmlx_engine/model_configs.py` registers `<｜latest_reminder｜>` as an EOS token
so a hallucinated reminder marker terminates instead of leaking
(`tests/test_model_config_registry.py::test_deepseek_v4_eos_includes_latest_reminder`).

## 2. Roles

`system`, `user`, `assistant`, `tool`, `latest_reminder`, `developer`.

- `developer` is used only by DeepSeek's internal search-agent pipeline. The
  official API rejects it; we pass it through unchanged when a caller sends it.
- `latest_reminder` is **caller-supplied**. The adapter never invents one and
  never rewrites `system` into `latest_reminder`
  (`tests/test_dsv4_contract_hardening.py::test_dsv4_official_encoder_adapter_preserves_message_order`).

## 3. Thinking rail

| API input | encoder call | prompt suffix |
|---|---|---|
| `enable_thinking=False` | `thinking_mode="chat"` | ends with `</think>` |
| `enable_thinking=True`, `reasoning_effort="high"` | `thinking_mode="thinking"` + effort | model opens `<think>…</think>` |
| `reasoning_effort="max"` | `thinking_mode="thinking"` + `max` | extra system hint |

`reasoning_effort` is realized purely as a **text prefix prepended before the
system message**. The 0731 encoder exposes `low` (default), `high`, `max`.
Efforts are validated against the *selected bundle's*
`REASONING_EFFORT_PROMPTS`, never against a family-wide alias — older bundles
may expose a different subset.

`reasoning_effort` has no effect in chat mode.

> **Prefix-cache consequence.** Changing `reasoning_effort` mid-conversation
> changes byte 0 of the prompt. It is a total cache invalidation by
> construction, not a bug.

## 4. Tool calling (DSML)

Tools are declared on the **system or developer message** via its `tools` field
(OpenAI shape). The encoder then injects a `## Tools` block containing the DSML
grammar explanation followed by `### Available Tool Schemas` and one JSON
schema per line.

A call looks like:

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="function_name">
<｜DSML｜parameter name="param" string="true">string_value</｜DSML｜parameter>
<｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜>
```

- `string="true"` → raw string value.
- `string="false"` → JSON value (number, boolean, array, object).

Tool results are wrapped in `<tool_result>` inside **user** messages, and when
several are present they are sorted by the order of the corresponding
`tool_calls` in the preceding assistant message
(`sort_tool_results_by_call_order`, `merge_tool_messages`).

### 4.1 `function.arguments` must be a JSON string

The bundled encoder calls `json.loads()` on `function.arguments` itself. Passing
a `dict` makes it wrap the whole dict under a single DSML `arguments`
parameter, which corrupts both tool-history continuation and the DSML
exemplars. The adapter normalizes dict → JSON string **only for DSV4**; the dict
form stays correct for Qwen/Mistral/Llama templates.

### 4.2 The prompt tool catalog is turn-independent — INVARIANT

`prompt_tool_catalog()` returns the **full authorized tool set**, in caller
order, with duplicate names dropped. It must never depend on the current turn.

Before v1.6.21 the catalog was narrowed to the tool named by the latest user
turn, falling back to the tool named in the most recent assistant `tool_calls`.
That produced two defects:

1. **Prefix-cache collapse.** An encoder-only probe over a five-turn agentic
   transcript pinned reuse at 918 chars — the first schema byte — as soon as
   the agent switched tools (51.9% then 44.4% reuse), while a stable catalog
   reused 100% on every turn. Every agent iteration cold-prefilled the whole
   growing transcript.
2. **Mid-loop capability loss.** After the agent called `write_file`, the
   rendered catalog contained *only* `write_file`. `read_file` and
   `run_command` were authorized by the API but invisible to the model.

`request_explicitly_requests_tool()` survives for *intent* detection, which
drives execution decisions and never touches the prompt prefix.

### 4.3 Fallback injection must stay off for canonical prompts

`vmlx_engine/api/tool_calling.py` will prepend a synthetic tool contract when a
template drops schemas. For DSV4 this must never fire on canonical encoder
output — a duplicated contract is the documented cause of an unbounded literal
`response` loop.

`_dsv4_has_exact_native_tool_contract()` gates on
`tokenizer._vmlx_dsv4_chat_template_shim`, which
`vmlx_engine/loaders/load_jangtq_dsv4.py:1537` sets on the production
tokenizer. **Any harness that constructs its own tokenizer stub must set that
attribute**, or it will observe a fallback injection that production does not
perform.

The secondary path `_dsv4_has_scoped_history_examples` accepts a DSV4-shaped
prompt that carries concrete DSML history but no `## Tools` block. It requires
**one** exemplar, not one per authorized tool: with a stable broad catalog only
the tools that actually ran can ever appear as exemplars.

## 5. Multi-turn reasoning retention

`drop_thinking` (default `True`) strips `<think>…</think>` from assistant turns
*before* the last user message.

**With tools present on any message, the bundle encoder force-disables
`drop_thinking`** (`encoding_dsv4.py`: `if any(m.get("tools") for m in
full_messages): effective_drop_thinking = False`). Tool-calling conversations
retain every turn's reasoning so the model can track multi-step work.

> **Prefix-cache consequence.** Enabling or disabling tools mid-conversation
> re-renders all prior assistant turns. That is an inherent full invalidation,
> not a defect. Toggling the tools switch mid-chat will always cold-prefill.

The adapter honours `jang_config.chat.reasoning.drop_earlier_reasoning` when the
caller does not override it.

## 6. Cache identity rendering (`add_generation_prompt=False`)

The official 0731 encoder has no `add_generation_prompt` argument. Its
`render_message` appends exactly one terminal rail after the final
user/developer message (or after a typed task). The adapter removes **only that
owned suffix** and never search/replaces an earlier assistant rail.

- Last message is `user`/`developer` → suffix is
  `ASSISTANT_SP_TOKEN + (thinking_start_token | thinking_end_token)`.
- Last message carries a `task` → suffix is the task token (with the assistant
  rail for `action`).
- Otherwise (for example a trailing `latest_reminder`) there is **no owned
  terminal rail**, so the full prompt is kept as its cache identity. A
  `latest_reminder` renders *after* the preceding user's assistant rail, making
  that rail non-terminal; it cannot be represented by the cache layer's
  final-N-token stripping contract.

If the declared suffix is not present at the end of the rendered prompt the
adapter raises rather than guessing.

## 7. Quick-instruction task tokens

Appended via a message's `"task"` field for single-token / short-form outputs.

| Token | Purpose | Placement |
|---|---|---|
| `<｜action｜>` | search-vs-answer routing | after assistant prefix + thinking token |
| `<｜title｜>` | conversation title | after the assistant's EOS |
| `<｜query｜>` | search-query generation | after user content |
| `<｜authority｜>` | source-authority demand | after user content |
| `<｜domain｜>` | prompt domain | after user content |
| `<｜extracted_url｜>` / `<｜read_url｜>` | per-URL fetch decision | after user content |

`DS_TASK_SP_TOKENS` on the bundle encoder is the authority for these strings;
the adapter raises if a requested task has no exposed suffix.

## 8. Completion parsing

`parse_completion()` mirrors the streaming parser for batch callers.

The generation loop consumes the stop token instead of returning it in decoded
text, but the official parser requires an ordinary visible completion to end
with the encoder's EOS. Tool-call-only turns hid this because their canonical
DSML close is itself terminal. The adapter restores the consumed stop token
before invoking the bundle parser, exactly as the production DSML parser does,
and does **not** convert a rejected grammar into an apparently valid answer —
callers need the failure so harnesses fail closed.

## 9. Long-context mode (`DSV4_LONG_CTX`)

- `1` is the supported runtime mode. `Model.make_cache()` returns
  `DeepseekV4Cache` on `compress_ratio>0` layers (CSA/HCA + SWA composite) and a
  bounded `RotatingKVCache` on ratio-zero local SWA layers, preserving the
  native 128-token ring geometry.
- The paged prefix cache uses a dedicated `deepseek_v4` block record with
  `deepseek_v4_v10_delta` metadata. v9 keys DSV4 prompt cache blocks at N-1
  tokens so the last prompt token is re-fed on prefix hits. The loader installs
  the prefill mask-trim patch required for prompts beyond the sliding window.
- `cache_salt` / `skip_prefix_cache` still bypass every cache layer for
  benchmarks. DSV4 is no longer force-bypassed by family.
- Generic TurboQuant KV must stay **off** for DSV4: the composite cache *is* the
  cache-size strategy, and layering generic TQ-KV would double-quantize the
  compressed CSA/HCA latents.

---

## Reproducing the prefix-cache proof

`prefix_probe` renders each turn's cache identity with the real bundle encoder
and reports the longest common prefix between consecutive turns. It needs no
model load — only the bundle's `encoding/` directory.

A healthy result is `PREFIX-OK` on every consecutive pair: identity(turn N)
must be a strict prefix of identity(turn N+1). Anything less means the tokens
after the divergence point are re-prefilled on every iteration.
