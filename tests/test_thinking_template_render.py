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
from vmlx_engine.utils.chat_template_kwargs import (
    build_chat_template_kwargs,
    ensure_thinking_off_sentinel,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (used by Layer 1 and Layer 2 and audit-time bisection scripts)
# ---------------------------------------------------------------------------


def _render_with_tokenizer(
    model: ThinkingTemplateModel, *, enable_thinking: bool
) -> str:
    """Render the chat template the same way the engine does.

    Engine prompt-render path (see vmlx_engine/engine/simple.py:389-446 and
    :687-752):

    1. Default path: ``tokenizer.apply_chat_template`` from transformers.
    2. DSV4-Flash bundles ship ``encoding/encoding_dsv4.py`` instead of a
       Jinja template; tokenizer_config.json has no ``chat_template`` key
       and ``apply_chat_template`` raises ValueError. The engine falls
       through to ``vmlx_engine.loaders.dsv4_chat_encoder.apply_chat_template``.

    This helper mirrors that fallthrough so the audit verifies exactly
    what the engine renders. Tests that fail here are template bugs
    visible to the engine, not test bugs.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model.model_path), trust_remote_code=True
    )
    messages = [{"role": "user", "content": model.sample_user_message}]
    tpl_kwargs = build_chat_template_kwargs(
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        rendered = tok.apply_chat_template(messages, **tpl_kwargs)
        return rendered if isinstance(rendered, str) else str(rendered)
    except ValueError as exc:
        if "chat_template" in str(exc) and model.family == "deepseek_v4":
            from vmlx_engine.loaders.dsv4_chat_encoder import (
                apply_chat_template as dsv4_apply,
            )
            return dsv4_apply(
                messages,
                enable_thinking=enable_thinking,
                model_path=str(model.model_path),
            )
        raise


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


def test_minimax_thinking_off_adds_empty_thought_sentinel():
    """MiniMax-M2 direct rail needs an explicit closed empty thought.

    The native template's ``enable_thinking=False`` branch ends at the
    assistant marker. Live MiniMax-M2.7-JANGTQ_K then opens a visible
    ``<think>`` block on exact-answer prompts. The engine post-render contract
    appends ``<think>\n</think>`` for MiniMax only, matching the prompt variant
    that live-tested cleanly.
    """
    rendered = "]~b]ai\n"
    fixed = ensure_thinking_off_sentinel(
        rendered,
        family_name="minimax",
        model_name="MiniMax-M2.7-JANGTQ_K",
    )

    assert fixed.endswith("<think>\n</think>\n\n")
    assert _has_empty_think_pair(fixed)


def test_thinking_off_sentinel_closes_open_thought_with_stable_shape():
    prompt = "]~b]ai\n<think>\n"

    fixed = ensure_thinking_off_sentinel(prompt, family_name="minimax")

    assert fixed.endswith("<think>\n</think>\n\n")
    assert _has_empty_think_pair(fixed)


def test_thinking_off_sentinel_does_not_touch_other_families_or_tools():
    prompt = "<|im_start|>assistant\n"

    assert (
        ensure_thinking_off_sentinel(prompt, family_name="qwen3_5")
        == prompt
    )
    assert (
        ensure_thinking_off_sentinel(
            prompt,
            family_name="minimax",
            tools_present=True,
        )
        == prompt
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
    """Load the model via SimpleEngine and produce up to max_tokens of text.

    SimpleEngine's `generate` takes a pre-rendered `prompt: str` and is
    async. We render the chat template here (mirrors what the engine
    request handler does) and run the async call via asyncio.run.
    """
    import asyncio

    from vmlx_engine.engine.simple import SimpleEngine

    rendered = _render_with_tokenizer(model, enable_thinking=enable_thinking)

    async def _run() -> str:
        eng = SimpleEngine(model_name=str(model.model_path))
        await eng.start()
        try:
            output = await eng.generate(
                prompt=rendered,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                enable_thinking=enable_thinking,
            )
        finally:
            try:
                await eng.stop()
            except Exception:
                logger.warning(
                    "stop failed for %s", model.arch_name, exc_info=True
                )
        # GenerationOutput exposes both `text` (cleaned) and `raw_text`
        # (pre-clean). For reasoning detection we want the raw form so
        # `<think>...</think>` tags survive clean_output_text.
        return getattr(output, "raw_text", None) or getattr(output, "text", "")

    return asyncio.run(_run())


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
