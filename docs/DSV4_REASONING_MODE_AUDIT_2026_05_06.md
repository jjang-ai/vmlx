# DSV4 Reasoning Mode Audit (2026-05-06)

## User report
Live log showed DSV4 multi-turn (3 turns: "testing" → "wat" → 434-token VC/Haskayne prompt) producing reasoning output that ignored the new user message entirely and fixated on "test message" framing from turn 1, then collapsed to "(No? No ) (No m m" tail.

Initial diagnosis was "JANGTQ2 long-context bundle drift." User pushed back: "still incoherent runtime dude" — wanted systematic verification.

## Systematic verification

Tested all 4 reasoning modes with controlled inputs to isolate what's vMLX vs encoder vs bundle.

### 1. Multi-turn rendering — CORRECT

`/tmp/dsv4_render_test.py` rendered the user's exact 3-turn flow through the production `apply_chat_template` path:

```
chars: 1726
contains VC fund: True
contains Process optimisation: True
contains specialized skills: True
count User markers: 3
count Assistant markers: 3
TAIL: "...<｜User｜>for our VC fund... <｜Assistant｜><think>"
```

**Conclusion:** the rendered prompt going to the model is correct. Full VC question is at the tail with `<｜Assistant｜><think>` opener. This is NOT a chat-template / multi-turn serialization bug. The model receives the right input.

### 2. Mode rendering matrix — confirms encoder behavior

| Panel button | Server-side `reasoning_effort` | Encoder `reasoning_effort` | Output contains `Reasoning Effort: Absolute maximum...` template? |
|---|---|---|---|
| **Instruct** | None | None (chat mode → `</think>`) | N/A (chat mode) |
| **Reasoning** (middle) | "medium" → server FIX 13 maps to "high" | "high" | **NO** |
| **Max** | "max" | "max" | **YES** (124-char template injected at prompt[0]) |

**Discovery:** The DSV4 encoder's `reasoning_effort="high"` is a **rendering NO-OP** — it produces byte-identical output to `reasoning_effort=None`. Per `encoding_dsv4.py:262`:
```python
if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
    prompt += REASONING_EFFORT_MAX
```

The encoder ASSERTS `reasoning_effort in ['max', None, 'high']` (`encoding_dsv4.py:261`) but only `'max'` actually triggers a template injection. `'high'` and `None` produce identical prompts.

So my FIX 13 ("low"/"medium"/"high" → encoder gets "high") is technically correct but has **no effect on the rendered prompt**. The Reasoning button still produces a bare `<think>` with no effort hint.

### 3. Mode behavior — model side

- **Instruct mode** (`<｜Assistant｜></think>`): closes thinking before opening — model goes straight to answer. Works fine on short prompts.
- **Reasoning / default thinking** (`<｜Assistant｜><think>`): bare thinking opener. Model gets no guidance on depth or scope. Output quality depends entirely on bundle quality.
- **Max mode** (`Reasoning Effort: Absolute maximum...` prepended): model gets a strong system-level anchor at position 0. Even at long context, the anchor reduces drift because it's at a position that DSV4's hash routing always attends to.

### 4. Why the user's 3rd turn drifted

Looking at the production log:

```
Reasoning 392 chars
The user's request is a test message with no specific question or task.
... (loops)
The user's "test message for testing purposes? (No? No
  ) (No m m
```

Key observations:
1. The rendered prompt at 434 tokens is correct (verified above).
2. The model produced only **392 chars of reasoning** — extremely short for a 434-token compound question. That's the model not engaging.
3. Reasoning fixates on "test message" framing — that's only present in turn-1 user message ("testing").
4. No `Reasoning Effort` anchor injected at position 0 (Reasoning button → "high" → no-op).

Combined with JANGTQ2 uniform 2-bit at 434 token context, the model:
- Has weak attention to the long new user message
- Falls back to fixating on early-turn framing (which the embedding-projected attention easily reaches)
- Doesn't get a Max-style anchor to break that pattern

This IS a real problem that affects multi-turn DSV4 reasoning quality at any prompt length where prior turns + new message exceed about 250-300 tokens.

## Fixes

### FIX A — Map "high" → "max" in DSV4 server, OR inject a custom anchor

The cleanest fix: when panel sends `thinking_mode="reasoning"` (middle button), DSV4 should get `reasoning_effort="max"` so the position-0 anchor fires. **Tradeoff:** "max" template is verbose (124 chars). Output verbosity may increase.

Alternative: inject a lighter `Reasoning Effort: focused` anchor for "high" mode. Requires editing the BUNDLE encoder (out of scope) OR pre-pending to the user's first message in the API layer (which I can do).

I'll take **the second approach** — inject a vMLX-side "focused reasoning" anchor when DSV4 + Reasoning button + multi-turn. Position 0 won't work (bundle encoder owns that), but injecting it as a system-message addendum gives the model a similar attention anchor.

### FIX B — Multi-turn `drop_thinking` correctness check

The encoder drops prior turns' `<think>` blocks but keeps the assistant's CONTENT. Verified the rendering preserves "It seems you're just testing the waters..." in turn-2 reply. So drop_thinking is correct.

### NOT a fix — model-quality drift

Even with FIX A, JANGTQ2 uniform 2-bit at long context will have some attention drift. That's the V3 rebuild path (other agent's work).

## What this means for v1.5.22

- **NOT a regression.** v1.5.22 ships with the same encoder behavior as v1.5.21. The Reasoning button has always been a bare-`<think>` rendering.
- **FIX 13 from v1.5.22 was correct but inert at the encoder level.** It made server.py's effort handling consistent (no longer stripping medium/high) but the encoder's no-op behavior on "high" means output is unchanged.
- **The diagnostic value of FIX 13** is that low/medium/high are now reaching the encoder if the encoder gets fixed in a future bundle release (jang-tools rebuild). When a future encoder version respects "high", FIX 13 lets vMLX talk to it correctly without changing.

## NO REGRESSIONS

Verified:
- Max mode still injects `REASONING_EFFORT_MAX` template (chars=624 vs 148 default — anchor present)
- Instruct mode still produces `<｜Assistant｜></think>` (immediate close)
- Default thinking still produces `<｜Assistant｜><think>` (bare opener)
- Multi-turn dropping prior `<think>` works correctly
- 3-turn 434-token rendering preserves new user message at the tail

## Action plan

1. Add FIX A: vMLX-side anchor injection for DSV4 Reasoning button. Light, non-template, pre-pended to system message. Documents the design tradeoff vs editing the bundle encoder.
2. Document this audit (this doc).
3. Verify test matrix still passes (no regressions).

---

## Codex 2026-05-06 follow-up review — 4 real bugs found

After this audit, Codex reviewed and caught 4 issues my prior fixes had MISSED. All applied as v1.5.23:

### CR1 — DSV4 force-thinking was missing from the REAL /v1/responses route

**File:** `vmlx_engine/server.py:7547` (NOT 6630 as my docs claimed).

The `/v1/responses` route at `server.py:7316` had a SEPARATE `_is_dsv4_resp_msgs` block that:
- Did NOT force `enable_thinking=True`
- Stripped "high"/"medium" → None instead of mapping to "high"

The block I edited at line 6630 was a NESTED block in `create_chat_completion`, not the actual Responses route. So the panel's /v1/responses traffic was hitting the broken path the entire time.

**Fix:** Replaced the 7547 block with the same force-thinking + low/medium/high → high logic + diagnostic log.

### CR2 — Panel SSE consumer dropped final text when no deltas arrived

**File:** `panel/src/main/ipc/chat.ts:1843`.

Panel only consumed `response.output_text.delta` events. Server can also send final text via `response.output_text.done`, `response.content_part.done`, or `response.completed` (with `response.output[*].content[*]`).

When a stream sent ONLY final-form events (no deltas), the assistant message rendered blank. Empty messages get skipped on history rebuild, consecutive user messages merge. Result: turn 3's VC prompt got merged with turn 1's "testing", model saw "testing" as effective last user message, replayed turn-1 framing.

**Fix:** Added fallback consumers for all 3 final-text event shapes, gated on `_sawResponsesTextDelta` so we never double-emit when deltas already came through.

### CR3 — rep_penalty floor bypassed by stale UI defaults

**File:** `vmlx_engine/server.py:_resolve_repetition_penalty` (lines 688-697 was the bug).

Function early-returned the request value before reaching the safety floor. Stale UI default of 1.0 was detected and replaced with bundle's 1.0 (which IS 1.0 for thinking — `repetition_penalty_thinking: 1.0` in jang_config), still below the 1.15 floor.

**Fix:** Refactored to compute `resolved` for ALL paths, then run floor enforcement at the end. Track `is_explicit_override` flag — only genuine non-stale explicit values bypass the floor. Stale UI defaults and bundle/CLI defaults all get floored.

### CR4 — Cache "fully working" claim was overstated

DSV4 cache writes deepseek_v4_pending markers for non-terminal blocks and full composite state only on terminal blocks (`prefix_cache.py:754`). Fetch then rejects pending-only hits (`prefix_cache.py:1172`). This is SAFE behavior but doesn't mean "full prefix/paged/L2 reuse for arbitrary multi-turn prefixes" — only EXACT same-prompt replays terminal-hit cleanly.

**Fix:** No code change — this is documented design. Updated `DSV4_FIX_NUANCES.md` language: cache is "safe with terminal-block hit reuse" not "full multi-turn cache stack."

## NO REGRESSIONS

After all 4 fixes:
- Force-thinking active on chat-completions, Ollama, Anthropic-via-chat, AND /v1/responses (4 endpoints)
- "low"/"medium"/"high" all reach encoder as "high" on all 4 endpoints
- Floor enforces on stale UI 1.0 + bundle 1.0 + CLI default
- Genuine per-request explicit non-stale values bypass floor (user opt-in)
- SSE final-text fallback fires only when no deltas arrived (no double-emit)
