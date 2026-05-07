"""Audit: per-arch thinking-template render and live-engine no-leak.

Two layers:

- Layer 1 (tokenizer-only): apply chat template via transformers
  AutoTokenizer with enable_thinking=True/False. Asserts no unclosed
  trailing <think> when enable_thinking=False. Fast — no Metal load.
- Layer 2 (live engine, marker `live`): load via SimpleEngine, generate
  up to 128 tokens with enable_thinking=False, assert no <think>...</think>
  block leaks into user-visible content.

Spec §8.3, §17.3. Plan
docs/superpowers/plans/2026-05-06-thinking-template-render-audit.md.
"""

from __future__ import annotations

import logging
from typing import Optional

import pytest

from tests.fixtures.thinking_template_models import (
    ALL_MODELS,
    AT_RISK_MODELS,
    SANITY_MODELS,
    ThinkingTemplateModel,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (used by Layer 1 and Layer 2 and audit-time bisection scripts)
# ---------------------------------------------------------------------------


def _render_with_tokenizer(
    model: ThinkingTemplateModel, *, enable_thinking: bool
) -> str:
    """Render the chat template via transformers AutoTokenizer.

    Mirrors the engine apply_chat_template kwargs path used in
    vmlx_engine/engine/simple.py:407 and :709.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model.model_path), trust_remote_code=True
    )
    messages = [{"role": "user", "content": model.sample_user_message}]
    rendered = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return rendered if isinstance(rendered, str) else str(rendered)


def _has_unclosed_open_think(prompt: str) -> bool:
    """True iff prompt has a trailing <think> with no following </think>."""
    last_open = prompt.rfind("<think>")
    if last_open < 0:
        return False
    after = prompt[last_open + len("<think>") :]
    return "</think>" not in after


def _has_empty_think_pair(prompt: str) -> bool:
    """True iff prompt contains a closed empty <think>...</think> pair.

    Tolerates whitespace / newlines between the open and close tags so
    `<think>\\n</think>`, `<think>\\n\\n</think>`, etc. all count.
    """
    if "<think>" not in prompt or "</think>" not in prompt:
        return False
    last_open = prompt.rfind("<think>")
    after = prompt[last_open + len("<think>") :]
    close_idx = after.find("</think>")
    if close_idx < 0:
        return False
    inner = after[:close_idx]
    return inner.strip() == ""


# ---------------------------------------------------------------------------
# Layer 1: tokenizer-only render assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in AT_RISK_MODELS],
)
def test_at_risk_template_honors_enable_thinking_false(
    model: ThinkingTemplateModel,
):
    """Layer 1 — at-risk archs (think_in_template=False).

    With enable_thinking=False the chat template must not emit a trailing
    unclosed <think>. Acceptable shapes: closed empty pair (template
    self-honors), no <think> at all, or a closed <think>...</think> with
    template-emitted content.

    Failure here means the template needs a per-bundle patch (the engine
    inject that pre-94b16d22 covered for it has been removed).
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    rendered = _render_with_tokenizer(model, enable_thinking=False)

    assert not _has_unclosed_open_think(rendered), (
        f"{model.arch_name}: chat template emitted an unclosed <think> with "
        f"enable_thinking=False — model will auto-think.\n"
        f"--- rendered prompt ---\n{rendered}\n--- end ---"
    )


@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in SANITY_MODELS],
)
def test_sanity_template_honors_enable_thinking_false(
    model: ThinkingTemplateModel,
):
    """Layer 1 — sanity archs (think_in_template=True).

    Same invariant as at-risk; expected to pass without intervention.
    Failure here means the registry's think_in_template=True is wrong for
    this family OR the bundle's template behavior diverges from the
    family default.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    rendered = _render_with_tokenizer(model, enable_thinking=False)

    assert not _has_unclosed_open_think(rendered), (
        f"{model.arch_name}: sanity-arch template emitted an unclosed "
        f"<think> with enable_thinking=False.\n"
        f"--- rendered prompt ---\n{rendered}\n--- end ---"
    )


@pytest.mark.parametrize(
    "model",
    [pytest.param(m, id=m.arch_name) for m in ALL_MODELS],
)
def test_template_renders_nonempty_when_enabled(
    model: ThinkingTemplateModel,
):
    """Layer 1 — sanity check on the enable_thinking=True path.

    A chat-template patch fixing the False path must not break the True
    path. We only require a non-empty rendered prompt; presence/absence
    of <think> in the prompt is template-specific (some emit it eagerly,
    others rely on the model to open it). Layer-2 thinking-on coverage
    is in test_live_at_risk_thinking_present_when_enabled below.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    rendered = _render_with_tokenizer(model, enable_thinking=True)

    assert rendered.strip(), (
        f"{model.arch_name}: chat template returned empty prompt with "
        f"enable_thinking=True"
    )


# ---------------------------------------------------------------------------
# Layer 2: live-engine generation (slow, model load required, marker `live`)
# ---------------------------------------------------------------------------


def _live_generate(
    model: ThinkingTemplateModel,
    *,
    enable_thinking: bool,
    max_tokens: int = 128,
) -> str:
    """Load the model via SimpleEngine and produce up to max_tokens of text."""
    from vmlx_engine.engine.simple import SimpleEngine

    eng = SimpleEngine()
    eng.load_model(str(model.model_path), is_mllm=False)
    try:
        out = eng.generate(
            messages=[
                {"role": "user", "content": model.sample_user_message}
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            enable_thinking=enable_thinking,
        )
    finally:
        try:
            eng.unload()
        except Exception:
            logger.warning(
                "unload failed for %s", model.arch_name, exc_info=True
            )
    return out if isinstance(out, str) else str(out)


def _output_contains_thinking(output: str) -> bool:
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
    must not emit <think>...</think> in user-visible content.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    output = _live_generate(model, enable_thinking=False, max_tokens=128)

    assert not _output_contains_thinking(output), (
        f"{model.arch_name}: live generation with enable_thinking=False "
        f"produced thinking content.\n--- output ---\n{output}\n--- end ---"
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
    open a <think> block somewhere in the response. Pure correctness: the
    flag still routes correctly.
    """
    if not model.model_path.is_dir():
        pytest.skip(f"Model not present on disk: {model.model_path}")

    output = _live_generate(model, enable_thinking=True, max_tokens=256)

    assert "<think>" in output, (
        f"{model.arch_name}: enable_thinking=True did not produce any "
        f"<think> tag.\n--- output ---\n{output}\n--- end ---"
    )
