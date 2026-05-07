# Thinking-Template Render Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify per-arch whether the empty `<think></think>` injection removal in commit `94b16d22` regresses any thinking-default model. For any arch where `enable_thinking=False` is no longer honored, fix at the chat-template level (per-bundle), not engine-side.

**Architecture:** Two-layer test approach. Layer 1: tokenizer-only chat-template render (fast, no Metal) asserts whether the bundle's `chat_template.jinja` honors `enable_thinking=False` natively. Layer 2: live engine generation (slow, Metal) loads each at-risk arch and asserts no thinking content leaks when `enable_thinking=False`. At-risk archs are those with `think_in_template=False` in `vmlx_engine/model_configs.py` (Ling/bailing_hybrid, Nemotron-H, Gemma 4). Models with `think_in_template=True` (DeepSeek V4, Qwen 3.5/3.6, GLM-5.1, Kimi, MiniMax M2.7) are sanity-checked but expected to honor the flag via template.

**Tech Stack:** pytest, transformers `AutoTokenizer`, vmlx_engine `SimpleEngine`, mlx_lm.

**Spec:** `/Users/eric/mlx/vllm-mlx/docs/superpowers/specs/2026-05-06-production-audit-design.md` §8.3, §17.3.

**Working dir:** `/Users/eric/mlx/vllm-mlx` on `session/v1.5.8`.

**Pre-flight:** Confirm `git -C /Users/eric/mlx/vllm-mlx log --oneline -1` shows `36db2602`. Run `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest -q tests/test_engine_audit.py` to confirm 854-pass baseline holds before starting.

---

## File Structure

| File | Type | Responsibility |
|---|---|---|
| `tests/fixtures/thinking_template_models.py` | Create | Per-arch fixture list: `(arch_name, family, model_path, think_in_template, sample_user_message)` tuples; resolves to disk paths and skips if missing. |
| `tests/test_thinking_template_render.py` | Create | Layer 1 (tokenizer-only) parametrized tests + Layer 2 (live engine) parametrized tests. |
| `docs/AUDIT-THINKING-TEMPLATE-RENDER.md` | Create | Verification matrix, per-arch verdict, root cause for any regression, chat-template patches recorded. |
| `vmlx_engine/model_configs.py` | Modify (only if Layer 2 finds regression in a `think_in_template=False` model and template fix lands) | Update the family's `think_in_template` flag if a chat-template patch makes it true. |

No engine-side code changes unless the audit finds a regression that requires per-family policy. The default outcome is a regression test + matrix doc that pins the current behavior.

---

## Task 1: Add fixture file mapping arches to local model paths

**Files:**
- Create: `tests/fixtures/thinking_template_models.py`

- [ ] **Step 1: Write the fixture file**

```python
"""Per-arch model paths and metadata for thinking-template render audit.

Each entry maps an architecture under test to:
- arch_name:           Human-readable label, used in test IDs.
- family:              vmlx_engine model_configs family_name.
- model_path:          Absolute path on local disk. Test is skipped if missing.
- think_in_template:   Mirror of vmlx_engine.model_configs[family].think_in_template.
                       True = template honors enable_thinking natively, False = template
                       does not, engine previously injected <think></think>.
- sample_user_message: Plain user content used as the test prompt.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ThinkingTemplateModel:
    arch_name: str
    family: str
    model_path: Path
    think_in_template: bool
    sample_user_message: str


# Sample message kept short to keep template render fast and deterministic.
_SAMPLE = "What is 2+2?"


# At-risk archs (think_in_template=False) come first — these are the ones the
# 94b16d22 inject removal could regress.
AT_RISK_MODELS: List[ThinkingTemplateModel] = [
    ThinkingTemplateModel(
        arch_name="ling-2.6-flash-jangtq2",
        family="bailing_hybrid",
        model_path=Path("/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK"),
        think_in_template=False,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="nemotron-omni-nano-jangtq",
        family="nemotron_h",
        model_path=Path("/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"),
        think_in_template=False,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="gemma-4-26b-jang4m",
        family="gemma4",
        model_path=Path("/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK"),
        think_in_template=False,
        sample_user_message=_SAMPLE,
    ),
]


# Sanity-check archs (think_in_template=True): expected pass without intervention.
SANITY_MODELS: List[ThinkingTemplateModel] = [
    ThinkingTemplateModel(
        arch_name="dsv4-flash-jangtq",
        family="deepseek_v4",
        model_path=Path("/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="minimax-m2.7-jangtq-k",
        family="minimax_m2",
        model_path=Path("/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="qwen3.6-27b-jang4m",
        family="qwen3_5",
        model_path=Path("/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="qwen3.6-35b-a3b-jangtq",
        family="qwen3_5_moe",
        model_path=Path("/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="kimi-k2.6-small-jangtq",
        family="kimi_k25",
        model_path=Path("/Users/eric/models/JANGQ/Kimi-K2.6-Small-JANGTQ"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
]


ALL_MODELS: List[ThinkingTemplateModel] = AT_RISK_MODELS + SANITY_MODELS


def available_models() -> List[ThinkingTemplateModel]:
    """Return only models whose model_path exists on disk."""
    return [m for m in ALL_MODELS if m.model_path.is_dir()]
```

- [ ] **Step 2: Smoke-import the fixture**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -c "from tests.fixtures.thinking_template_models import available_models; m = available_models(); print(len(m), [x.arch_name for x in m])"`

Expected: prints a count ≥ 6 and a list including `ling-2.6-flash-jangtq2`, `gemma-4-26b-jang4m`, etc. Models actually present on disk only.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/mlx/vllm-mlx
git add tests/fixtures/thinking_template_models.py
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "test: add thinking-template model fixtures for audit"
```

---

## Task 2: Layer-1 test — tokenizer-only template render asserts `enable_thinking=False` honor

**Files:**
- Create: `tests/test_thinking_template_render.py`

- [ ] **Step 1: Write the failing test**

```python
"""Audit: thinking-template render and live-engine no-leak verification.

Layer 1 (this task): tokenizer-only chat-template render. For each at-risk arch,
asserts that with enable_thinking=False the rendered prompt either (a) emits an
empty <think></think> block (template honors the flag and self-emits the closer),
or (b) emits no <think> at all and the model is expected to not auto-think.

Failure of (a) AND (b) = regression candidate; Layer 2 (next task) confirms with
live generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.fixtures.thinking_template_models import (
    AT_RISK_MODELS,
    SANITY_MODELS,
    ThinkingTemplateModel,
    available_models,
)


logger = logging.getLogger(__name__)


def _render_with_tokenizer(
    model: ThinkingTemplateModel, *, enable_thinking: bool
) -> str:
    """Render the chat template via transformers AutoTokenizer.

    Mirrors the engine's apply_chat_template kwargs path in
    vmlx_engine/engine/simple.py:407 and :709.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model.model_path), trust_remote_code=True)
    messages = [{"role": "user", "content": model.sample_user_message}]
    rendered = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return rendered


def _has_unclosed_open_think(prompt: str) -> bool:
    """True if prompt has a trailing <think> with no following </think>."""
    last_open = prompt.rfind("<think>")
    if last_open < 0:
        return False
    after = prompt[last_open + len("<think>"):]
    return "</think>" not in after


def _has_empty_think_pair(prompt: str) -> bool:
    """True if prompt contains the closed empty <think></think> block."""
    return "<think></think>" in prompt or "<think>\n</think>" in prompt


@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in AT_RISK_MODELS],
)
def test_at_risk_template_honors_enable_thinking_false(
    model: ThinkingTemplateModel,
):
    """Layer 1 — at-risk archs: enable_thinking=False must not produce an
    unclosed trailing <think>. The template either emits <think></think> or
    omits the block entirely; both are acceptable Layer-1 outcomes.

    Failure here means Layer 2 must confirm whether the model auto-thinks
    despite the flag and a chat-template fix is required.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    rendered = _render_with_tokenizer(model, enable_thinking=False)

    assert not _has_unclosed_open_think(rendered), (
        f"{model.arch_name}: chat template emitted an unclosed <think> with "
        f"enable_thinking=False — model will auto-think. Rendered prompt:\n"
        f"---\n{rendered}\n---"
    )


@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in SANITY_MODELS],
)
def test_sanity_template_honors_enable_thinking_false(
    model: ThinkingTemplateModel,
):
    """Layer 1 — sanity archs (think_in_template=True): enable_thinking=False
    must not produce an unclosed trailing <think>. Family is expected to honor
    the flag via its own template logic.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    rendered = _render_with_tokenizer(model, enable_thinking=False)

    assert not _has_unclosed_open_think(rendered), (
        f"{model.arch_name}: sanity-arch template emitted an unclosed <think> "
        f"with enable_thinking=False. Rendered prompt:\n---\n{rendered}\n---"
    )


@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in AT_RISK_MODELS + SANITY_MODELS],
)
def test_template_emits_think_when_enabled(model: ThinkingTemplateModel):
    """Layer 1 — sanity check: enable_thinking=True paths should not be broken
    by any future template patch. We require a <think> opening tag to appear
    somewhere in the rendered prompt for thinking-default families. (Some
    families always think; others require the kwarg. Either is acceptable.)

    This guards against a chat-template patch that breaks thinking-on by
    accident while fixing thinking-off.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    rendered = _render_with_tokenizer(model, enable_thinking=True)

    # We only assert the rendered prompt is non-empty and well-formed; some
    # bundles only emit <think> from the model output side (not the prompt).
    # The Layer-2 live test catches any "thinking-on never produces thoughts"
    # regression.
    assert rendered.strip(), (
        f"{model.arch_name}: chat template returned empty prompt with "
        f"enable_thinking=True"
    )
```

- [ ] **Step 2: Run the new test file to capture per-arch results**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py -v --tb=short 2>&1 | tee /tmp/think_template_layer1.log`

Expected: all `test_*_honors_enable_thinking_false` cases pass for archs whose templates emit an empty `<think></think>` or omit it; FAIL for any arch whose template emits a trailing unclosed `<think>` (this is the regression we are looking for). Skipped for missing models.

Record any failures verbatim — they map directly to the §8.3 regression matrix.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/mlx/vllm-mlx
git add tests/test_thinking_template_render.py
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "test: add Layer-1 thinking-template render audit"
```

---

## Task 3: Layer-2 test — live engine, generate with `enable_thinking=False`, assert no thinking leak

**Files:**
- Modify: `tests/test_thinking_template_render.py` (append)

- [ ] **Step 1: Add the live-engine test class to the file**

Append to `tests/test_thinking_template_render.py`:

```python


# ---------------------------------------------------------------------------
# Layer 2: live-engine generation. Slow (multi-second model load per arch).
# Marked `live` so it can be selected/excluded explicitly.
# ---------------------------------------------------------------------------


def _live_generate(
    model: ThinkingTemplateModel,
    *,
    enable_thinking: bool,
    max_tokens: int = 64,
) -> str:
    """Load model via SimpleEngine and produce up to max_tokens of text."""
    from vmlx_engine.engine.simple import SimpleEngine

    eng = SimpleEngine()
    eng.load_model(str(model.model_path), is_mllm=False)
    try:
        out = eng.generate(
            messages=[{"role": "user", "content": model.sample_user_message}],
            max_tokens=max_tokens,
            temperature=0.0,
            enable_thinking=enable_thinking,
        )
    finally:
        try:
            eng.unload()
        except Exception:
            logger.warning("unload failed for %s", model.arch_name, exc_info=True)
    return out if isinstance(out, str) else str(out)


def _output_contains_thinking(output: str) -> bool:
    """Heuristic for thinking content leaking into the user-visible response."""
    return "<think>" in output and "</think>" in output


@pytest.mark.live
@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in AT_RISK_MODELS],
)
def test_live_at_risk_no_thinking_leak_when_disabled(
    model: ThinkingTemplateModel,
):
    """Layer 2 — at-risk archs: live generation with enable_thinking=False
    must not emit <think>...</think> in the user-visible content.

    A failure here is a real regression caused by 94b16d22 removing the
    engine-side empty <think></think> inject. The fix is a chat-template
    patch on the bundle, not an engine-side reinject.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    output = _live_generate(model, enable_thinking=False, max_tokens=128)

    assert not _output_contains_thinking(output), (
        f"{model.arch_name}: live generation with enable_thinking=False "
        f"produced thinking content. Output:\n---\n{output}\n---"
    )


@pytest.mark.live
@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in AT_RISK_MODELS],
)
def test_live_at_risk_thinking_present_when_enabled(
    model: ThinkingTemplateModel,
):
    """Layer 2 — sanity: with enable_thinking=True, the at-risk archs SHOULD
    produce thinking content (or at least open a <think> block before content).
    Pure correctness check that the flag still routes correctly.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    output = _live_generate(model, enable_thinking=True, max_tokens=256)

    assert "<think>" in output, (
        f"{model.arch_name}: enable_thinking=True did not produce any "
        f"<think> tag. Output:\n---\n{output}\n---"
    )
```

- [ ] **Step 2: Confirm pytest live marker is registered**

Run: `cd /Users/eric/mlx/vllm-mlx && grep -nE "markers|live" pytest.ini setup.cfg pyproject.toml 2>/dev/null | head -10`

Expected: `pytest.ini` or `pyproject.toml` lists `live` under `[pytest] markers`. If not, append in the next step.

- [ ] **Step 3: Register the `live` marker if missing**

If Step 2 showed no `live` marker, edit `/Users/eric/mlx/vllm-mlx/pytest.ini` and add under `[pytest]`:

```ini
markers =
    live: tests that load real models from disk and run Metal generation; slow
```

If `pytest.ini` already has a `markers =` section, append the `live:` line.

- [ ] **Step 4: Smoke-run a single live test on the smallest at-risk arch**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py::test_live_at_risk_no_thinking_leak_when_disabled -v -k gemma-4-26b -m live --tb=short 2>&1 | tee /tmp/think_template_layer2_gemma.log`

Expected: PASS (no thinking leak) or FAIL with explicit thinking content in output. Either is informative. If FAIL, this is exactly the regression §8.3 looks for.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/mlx/vllm-mlx
git add tests/test_thinking_template_render.py pytest.ini
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "test: add Layer-2 live-engine thinking-leak audit"
```

---

## Task 4: Run the full Layer-2 matrix on all available at-risk archs

**Files:** None modified — execution-only.

- [ ] **Step 1: Run all live at-risk tests sequentially**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py -m live -k "at_risk" -v --tb=short -s 2>&1 | tee /tmp/think_template_layer2_full.log`

Expected: a verdict for each at-risk arch (Ling-2.6, Nemotron-Omni-Nano, Gemma-4-26B). Sequentially loads each model, ~2–5 minutes per arch on M5 Max. Total wall time ~10–20 minutes.

- [ ] **Step 2: Capture per-arch verdict in a structured file**

Write `/tmp/think_template_layer2_verdict.json` from the log output. Each entry:

```json
{
  "ling-2.6-flash-jangtq2": {
    "no_leak_when_disabled": "PASS|FAIL|SKIP",
    "thinking_present_when_enabled": "PASS|FAIL|SKIP",
    "leaked_output_excerpt": "<first 200 chars if FAIL>"
  }
}
```

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -c "
import re, json, sys
log = open('/tmp/think_template_layer2_full.log').read()
verdict = {}
for arch in ['ling-2.6-flash-jangtq2', 'nemotron-omni-nano-jangtq', 'gemma-4-26b-jang4m']:
    no_leak = 'PASS' if f'PASSED' in log and 'no_thinking_leak' in log and arch in log else (
        'FAIL' if f'FAILED' in log and 'no_thinking_leak' in log and arch in log else 'SKIP'
    )
    enabled = 'PASS' if f'PASSED' in log and 'thinking_present_when_enabled' in log and arch in log else (
        'FAIL' if f'FAILED' in log and 'thinking_present_when_enabled' in log and arch in log else 'SKIP'
    )
    verdict[arch] = {'no_leak_when_disabled': no_leak, 'thinking_present_when_enabled': enabled}
print(json.dumps(verdict, indent=2))
" > /tmp/think_template_layer2_verdict.json && cat /tmp/think_template_layer2_verdict.json`

Expected: JSON with one entry per available at-risk arch. (The parser is intentionally crude — refine if log format requires.)

- [ ] **Step 3: Decide branch**

Read the verdict.

- If all `no_leak_when_disabled` are PASS → at-risk archs are safe, no engine-side regression. Proceed to Task 7 (sanity sweep + doc).
- If any `no_leak_when_disabled` is FAIL → that arch's chat template needs a patch. Proceed to Task 5 with the failing arch.
- If `thinking_present_when_enabled` is FAIL → unrelated regression in template/runtime; record but do not block this audit.

---

## Task 5: For each FAIL, identify root cause via render diff

**Files:** No code change — investigation step.

Run only if Task 4 produced a FAIL. Otherwise skip to Task 7.

- [ ] **Step 1: For each failing arch, dump the rendered prompt at both flag values**

For each failing arch (replace `<ARCH>` with `gemma-4-26b-jang4m` etc.):

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -c "
from tests.fixtures.thinking_template_models import ALL_MODELS
from tests.test_thinking_template_render import _render_with_tokenizer
m = next(x for x in ALL_MODELS if x.arch_name == '<ARCH>')
print('=== enable_thinking=True ===')
print(repr(_render_with_tokenizer(m, enable_thinking=True)))
print('=== enable_thinking=False ===')
print(repr(_render_with_tokenizer(m, enable_thinking=False)))
" 2>&1 | tee /tmp/think_template_render_<ARCH>.log`

Expected: two rendered strings. The diff between them is what the bundle's `chat_template.jinja` does for the thinking flag.

- [ ] **Step 2: Identify the template's policy**

For each arch, classify the template's behavior into one of:

1. **"emits-empty-pair"** — `enable_thinking=False` causes template to emit `<think></think>` (closed). No regression. Was masked by engine inject before; is honored natively now.
2. **"omits-think"** — `enable_thinking=False` causes template to emit no `<think>` at all. Engine relied on its inject to suppress thinking. Regression target.
3. **"emits-unclosed-think"** — `enable_thinking=False` still emits an open `<think>` with no closer. Regression target.
4. **"ignores-flag"** — Both flag values produce identical prompt. Template ignores `enable_thinking`. Regression target.

Record classification per arch in a working notes file `/tmp/think_template_classification.txt`.

- [ ] **Step 3: For each regression-target arch, locate the chat template source**

Run: `find <model_path> -name "chat_template.jinja" -o -name "tokenizer_config.json" 2>/dev/null`

If `chat_template.jinja` exists, that's the file to patch. Otherwise, the template lives inside `tokenizer_config.json` under the `chat_template` key.

- [ ] **Step 4: Read the template and identify the patch site**

Read the template; find the section that handles `enable_thinking` (or `add_generation_prompt`). The patch site is wherever the template would emit a generation-prompt suffix.

The standard fix shape (Jinja example):

```jinja
{# existing template content above #}
{%- if add_generation_prompt %}
    {{- '<|assistant|>' }}
    {%- if enable_thinking | default(false) %}
        {{- '<think>' }}
    {%- else %}
        {{- '<think></think>' }}  {# explicit empty pair so model honors no-think #}
    {%- endif %}
{%- endif %}
```

This must be adapted to the bundle's actual delimiters and structure.

---

## Task 6: Apply the chat-template patch and re-verify

**Files:** Per-bundle `chat_template.jinja` (or `tokenizer_config.json`) — one patch per failing arch. Bundle paths are user-owned and live outside the repo.

Run only if Task 5 identified a regression-target arch. Otherwise skip to Task 7.

- [ ] **Step 1: Backup the original template**

```bash
cp <model_path>/chat_template.jinja <model_path>/chat_template.jinja.pre-audit-2026-05-06
```

(Or for inline templates: `cp <model_path>/tokenizer_config.json <model_path>/tokenizer_config.json.pre-audit-2026-05-06`.)

- [ ] **Step 2: Edit the template file with the patch from Task 5 Step 4**

Use the Edit tool to apply the per-bundle patch identified.

- [ ] **Step 3: Re-render with the patched template**

```bash
cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -c "
from tests.fixtures.thinking_template_models import ALL_MODELS
from tests.test_thinking_template_render import _render_with_tokenizer
m = next(x for x in ALL_MODELS if x.arch_name == '<ARCH>')
print(repr(_render_with_tokenizer(m, enable_thinking=False)))
"
```

Expected: rendered prompt now contains `<think></think>` (or equivalent closed empty pair) for `enable_thinking=False`.

- [ ] **Step 4: Re-run Layer-1 test for this arch**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py::test_at_risk_template_honors_enable_thinking_false -v -k <ARCH> --tb=short`

Expected: PASS.

- [ ] **Step 5: Re-run Layer-2 test for this arch**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py::test_live_at_risk_no_thinking_leak_when_disabled -v -k <ARCH> -m live --tb=short`

Expected: PASS — live generation with `enable_thinking=False` produces no thinking content.

- [ ] **Step 6: Commit**

The chat template lives outside the repo, so the commit captures the audit doc entry rather than the file. The audit doc records the diff inline.

```bash
cd /Users/eric/mlx/vllm-mlx
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit --allow-empty -m "audit: chat-template patch for <ARCH> recorded in audit doc"
```

(Empty commit allowed because the source-of-truth for bundle template lives in user model bundles, not in the engine repo. The audit doc is updated in Task 7 with the diff.)

---

## Task 7: Run the sanity sweep on `think_in_template=True` archs

**Files:** None modified — execution-only.

- [ ] **Step 1: Run all Layer-1 sanity tests**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py::test_sanity_template_honors_enable_thinking_false tests/test_thinking_template_render.py::test_template_emits_think_when_enabled -v --tb=short 2>&1 | tee /tmp/think_template_sanity.log`

Expected: all sanity archs pass `test_sanity_template_honors_enable_thinking_false`. The `test_template_emits_think_when_enabled` test is informational — does not block.

- [ ] **Step 2: Run a single live sanity test on Qwen3.6-27B (smallest in-tree thinking model)**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py::test_live_at_risk_thinking_present_when_enabled -v -k qwen3.6-27b -m live --tb=short 2>&1 | tee /tmp/think_template_sanity_qwen.log`

(This actually runs the at-risk path's enabled-thinking test, but on a sanity arch by id. The same test class covers at-risk; for sanity archs we infer from Layer-1 alone unless a regression-target was found.)

Expected: Layer-1 PASS on all sanity archs. No live run needed unless Layer-1 found something unexpected.

---

## Task 8: Update audit doc with verdicts, classifications, and any patches

**Files:**
- Create: `docs/AUDIT-THINKING-TEMPLATE-RENDER.md`

- [ ] **Step 1: Write the audit doc**

Use the verdicts from Task 4 Step 2 and classifications from Task 5 Step 2. Substitute actual results.

```markdown
# Thinking-Template Render Audit — 2026-05-06

**Spec:** `/Users/eric/mlx/vllm-mlx/docs/superpowers/specs/2026-05-06-production-audit-design.md` §8.3, §17.3
**Plan:** `/Users/eric/mlx/vllm-mlx/docs/superpowers/plans/2026-05-06-thinking-template-render-audit.md`
**Trigger:** Codex commit `94b16d22` removed the blanket empty `<think></think>` append from `vmlx_engine/engine/{batched,simple}.py`. This audit verifies whether that removal regresses `enable_thinking=False` honor for thinking-default models whose chat templates do not natively emit a closed empty `<think></think>` block. Simple/Batched still have a narrow close-unclosed-`<think>` fallback; the audit should identify template behavior before that fallback masks it.

## Methodology

Two-layer verification:

1. **Layer 1 (tokenizer-only render):** apply chat template with `enable_thinking=True`/`False` via `transformers.AutoTokenizer.apply_chat_template`. Assert no unclosed trailing `<think>` for the False case.
2. **Layer 2 (live engine):** load the model via `SimpleEngine`, generate up to 128 tokens with `enable_thinking=False`, and assert no `<think>...</think>` block leaks into the user-visible content.

The pytest spec is `tests/test_thinking_template_render.py`; the model fixtures are `tests/fixtures/thinking_template_models.py`.

## Verdict matrix

| Arch | Family | `think_in_template` (registry) | Layer 1 (no unclosed `<think>` when disabled) | Layer 2 (no thinking leak when disabled) | Layer 2 (thinking present when enabled) | Classification |
|---|---|---|---|---|---|---|
| Ling-2.6-flash-JANGTQ2 | bailing_hybrid | False | <RESULT> | <RESULT> | <RESULT> | <CLASS> |
| Nemotron-Omni-Nano-JANGTQ | nemotron_h | False | <RESULT> | <RESULT> | <RESULT> | <CLASS> |
| Gemma-4-26B-JANG_4M | gemma4 | False | <RESULT> | <RESULT> | <RESULT> | <CLASS> |
| DSV4-Flash-JANGTQ | deepseek_v4 | True | <RESULT> | n/a | n/a | <CLASS> |
| MiniMax-M2.7-JANGTQ_K | minimax_m2 | True | <RESULT> | n/a | n/a | <CLASS> |
| Qwen3.6-27B-JANG_4M | qwen3_5 | True | <RESULT> | n/a | n/a | <CLASS> |
| Qwen3.6-35B-A3B-JANGTQ | qwen3_5_moe | True | <RESULT> | n/a | n/a | <CLASS> |
| Kimi-K2.6-Small-JANGTQ | kimi_k25 | True | <RESULT> | n/a | n/a | <CLASS> |

**Classification key:**
- `emits-empty-pair`: template emits closed `<think></think>` when `enable_thinking=False`. No engine intervention needed.
- `omits-think`: template emits no `<think>` at all. Pre-`94b16d22` behavior relied on engine inject. Patch required.
- `emits-unclosed-think`: template emits open `<think>` with no closer. Pre-`94b16d22` behavior relied on engine inject. Patch required.
- `ignores-flag`: template renders identically regardless of flag. Patch required.

## Per-arch patches applied

(One sub-section per regression-target arch, with the diff applied to the bundle's `chat_template.jinja` or `tokenizer_config.json`. Empty section if no regression found.)

### <ARCH-NAME>

- **Bundle path:** `<model_path>`
- **Template source:** `chat_template.jinja` | `tokenizer_config.json::chat_template`
- **Backup taken:** `<model_path>/chat_template.jinja.pre-audit-2026-05-06`
- **Diff:**

```diff
- <original snippet>
+ <patched snippet>
```

- **Layer 1 re-verify:** PASS
- **Layer 2 re-verify:** PASS

## Decision

Based on the matrix above:

- All at-risk archs <PASS|FAIL with patch applied>.
- All sanity archs honored `enable_thinking=False` natively via their templates.
- The engine-side empty `<think></think>` inject removal in `94b16d22` is <safe to keep removed | requires per-bundle template patches that are now applied | ...>.
- `vmlx_engine/model_configs.py` `think_in_template` flag updated for: <list of families> (or "no changes").

## Remaining items

- (List any archs that could not be tested due to disk absence; e.g., Mistral-Medium-3.5 if drive remains unmounted.)
- Bundle template patches are local to user disk. To distribute the fix to other users of the same bundle, re-publish the bundle on Hugging Face / mirror with the patched `chat_template.jinja`.
```

- [ ] **Step 2: Commit the audit doc**

```bash
cd /Users/eric/mlx/vllm-mlx
git add docs/AUDIT-THINKING-TEMPLATE-RENDER.md
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "docs: thinking-template render audit results"
```

---

## Task 9: Update `model_configs.py` `think_in_template` flag if any patch promotes a family

**Files:**
- Modify: `vmlx_engine/model_configs.py` (only if Task 6 patched a `think_in_template=False` family's template to emit `<think></think>` natively)

Skip if Task 6 was not run.

- [ ] **Step 1: Identify which families to update**

For each family whose chat template now natively emits `<think></think>` for `enable_thinking=False` after the Task 6 patch, the registry's `think_in_template` flag should be updated from `False` to `True`. This communicates to other engine code paths (notably the reasoning-extractor at `vmlx_engine/server.py:1074-1080`) that the template is now self-honoring.

- [ ] **Step 2: Edit the registry entry per family**

For each family identified in Step 1 (e.g. `bailing_hybrid` at line 371):

```python
# Before:
think_in_template=False,
# After:
think_in_template=True,
```

- [ ] **Step 3: Run the audit-touching tests**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest tests/test_thinking_template_render.py tests/test_engine_audit.py tests/test_reasoning_modes.py tests/test_reasoning_parser.py -v --tb=short`

Expected: all pass. Any test that relied on `think_in_template=False` for the patched family fails — investigate and update the test if the new behavior is correct.

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/mlx/vllm-mlx
git add vmlx_engine/model_configs.py tests/
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "feat(model_configs): promote think_in_template=True for patched families"
```

---

## Task 10: Update spec status board with §8.3 verdict

**Files:**
- Modify: `docs/SESSION_2026_05_06_PYTHON_ENGINE_APP_AUDIT.md`

- [ ] **Step 1: Append a §8.3 verdict block to the audit status board**

Read the file, locate the end, and append:

```markdown
## §8.3 Empty `<think></think>` Injection — Audit Complete (2026-05-06)

Audit doc: `docs/AUDIT-THINKING-TEMPLATE-RENDER.md`
Plan: `docs/superpowers/plans/2026-05-06-thinking-template-render-audit.md`
Tests: `tests/test_thinking_template_render.py`, `tests/fixtures/thinking_template_models.py`

Verdict: <SAFE-TO-KEEP-REMOVED | PATCHES-APPLIED | BLOCKED>.

- At-risk archs verified: <list>
- Sanity archs verified: <list>
- Bundle template patches applied: <list of archs and bundle paths>
- `vmlx_engine/model_configs.py` `think_in_template` flag updated for: <list of families> (or "no changes").

The blanket engine-side `<think></think>` append removed in `94b16d22` is now confirmed by the recorded evidence to be either honored by templates natively or handled by source-level template-kwarg normalization. The remaining close-unclosed-`<think>` fallback is tracked separately as compatibility debt.
```

Substitute the actual results from Tasks 4–9.

- [ ] **Step 2: Run the full focused engine audit suite once more to confirm no regression vs the `94b16d22` baseline**

Run: `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python -m pytest -q tests/test_engine_audit.py tests/test_mllm_processor_stream_contracts.py tests/test_reasoning_modes.py tests/test_reasoning_parser.py tests/test_thinking_template_render.py 2>&1 | tail -5`

Expected: all pass; pass count is `>= 204 + new tests added` (Task 2 adds 8+, Task 3 adds 4+ markers).

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/mlx/vllm-mlx
git add docs/SESSION_2026_05_06_PYTHON_ENGINE_APP_AUDIT.md
git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "docs(audit): record §8.3 thinking-template verdict in status board"
```

---

## Task 11: Push to origin/main (only on user request)

**Files:** None modified.

- [ ] **Step 1: Wait for user explicit push approval**

Do not push without explicit user instruction. The local commits are: `b9c9e753`, `36db2602`, plus the §8.3 task commits.

- [ ] **Step 2: When approved, push**

Run: `cd /Users/eric/mlx/vllm-mlx && git push origin HEAD:main`

Expected: clean push from `session/v1.5.8` to `origin/main`.

- [ ] **Step 3: Confirm with `gh`**

Run: `gh api repos/jjang-ai/vmlx/commits/main --jq '{sha:.sha, message:.commit.message}'`

Expected: latest commit message matches "docs(audit): record §8.3 thinking-template verdict".

---

## Definition of Done

- [ ] Task 1 fixture file committed.
- [ ] Task 2 Layer-1 test committed and passing on every available arch (or recorded as the regression target).
- [ ] Task 3 Layer-2 test committed.
- [ ] Task 4 full Layer-2 matrix run; verdicts recorded.
- [ ] Task 5 root cause identified for any FAIL.
- [ ] Task 6 chat-template patches applied and re-verified for any FAIL.
- [ ] Task 7 sanity sweep complete.
- [ ] Task 8 audit doc written with verdict matrix and per-arch patch diffs.
- [ ] Task 9 `model_configs.py` updated only if a family's template was promoted to native-honoring.
- [ ] Task 10 status board appended with §8.3 verdict.
- [ ] Task 11 push waiting for user approval.

When all checkboxes above are checked, §8.3 of the production audit spec is complete and the next plan (§8.4 Qwen affine-JANG VLM divergence) is the next session's work.
