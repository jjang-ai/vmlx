"""Shared chat-template kwarg normalization.

The public engine flag is ``enable_thinking``. Some shipped model templates use
the shorter ``thinking`` variable instead. Keep both variables in sync before
rendering so model templates do not need local, per-bundle edits just to honor
the standard API flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(("tokenize", "add_generation_prompt"))


def model_type_of(model: Any) -> str:
    """Best-effort ``model_type`` from a loaded model's config (dict or obj)."""
    config = getattr(model, "config", None)
    if isinstance(config, Mapping):
        return str(config.get("model_type", "") or "")
    return str(getattr(config, "model_type", "") or "")


def build_chat_template_kwargs(
    *,
    enable_thinking: bool | None,
    extra: Mapping[str, Any] | None = None,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    include_thinking_alias: bool = True,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Return safe kwargs for tokenizer/processor ``apply_chat_template``.

    ``enable_thinking`` is vMLX's canonical request-level control. If it is
    resolved, it wins over conflicting values inside ``extra`` and, by default,
    is mirrored to ``thinking`` for templates that use that variable name.
    ``tokenize`` and ``add_generation_prompt`` are reserved engine-owned values.
    """

    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }

    if enable_thinking is not None:
        thinking_value = bool(enable_thinking)
        kwargs["enable_thinking"] = thinking_value
        if include_thinking_alias:
            kwargs["thinking"] = thinking_value

    if extra:
        for key, value in extra.items():
            if key in RESERVED_CHAT_TEMPLATE_KWARGS:
                continue
            if enable_thinking is not None and key in ("enable_thinking", "thinking"):
                continue
            kwargs[key] = value

    if (
        include_thinking_alias
        and enable_thinking is None
        and "enable_thinking" in kwargs
        and "thinking" not in kwargs
    ):
        kwargs["thinking"] = bool(kwargs["enable_thinking"])

    # GLM-5.3 owns its native history default. Forcing clear_thinking=True
    # rewrites prior turns on every new user message and invalidates their
    # otherwise reusable causal cache prefix. Explicit caller values survive.
    if (
        model_type
        and model_type.lower().startswith("glm5")
        and model_type.lower() not in {"glm5_next", "glm5_next_text"}
    ):
        kwargs.setdefault("clear_thinking", True)

    return kwargs


# Dropped last, in this order, by a failing render's kwarg-strip retry: the tool schema matters more than any
# unknown extra kwarg, and the thinking switch more than the tool schema.
_STRIP_LAST_CHAT_TEMPLATE_KWARGS = ("tools", "thinking", "enable_thinking")


def template_kwarg_strip_order(kwargs: Mapping[str, Any]) -> list[str]:
    """Order in which a failing ``apply_chat_template`` retry drops kwargs, one at a time.

    Extra kwargs go first (most recently added first), then ``tools``, then the thinking switch;
    ``tokenize`` / ``add_generation_prompt`` are never dropped. Dropping ``tools`` before an extra kwarg
    that a template rejects (e.g. an effort value it asserts on) removed the native tool schema, so the
    model answered with a non-native tool call and the request failed tool parsing."""
    keys = [k for k in kwargs if k not in RESERVED_CHAT_TEMPLATE_KWARGS]
    extras = [k for k in reversed(keys) if k not in _STRIP_LAST_CHAT_TEMPLATE_KWARGS]
    return extras + [k for k in _STRIP_LAST_CHAT_TEMPLATE_KWARGS if k in kwargs]


def trailing_open_think_index(prompt: str) -> int:
    """Index of the generation prompt's open ``<think>`` rail, or -1.

    A template's thinking prefill is always the very end of the rendered
    prompt (``...assistant\n<think>\n``).  A ``<think>`` with any
    non-whitespace after it is CONTENT -- a user quoting reasoning tags, a
    pasted model transcript, source code -- never the rail.  The previous
    rule (last ``<think>`` anywhere with no ``</think>`` after it) truncated
    such prompts at the user's literal tag: measured on Qwen3.8 Flash-Next, a
    64,442-token prompt whose content held ``<think>`` at token 54,041 had its
    thinking-off render cut to 54,048 tokens, so the prefix-cache key and the
    reported prompt_tokens lost the last ~10.4k tokens, and every repeat
    re-prefilled them.
    """
    idx = prompt.rfind("<think>")
    if idx < 0 or prompt[idx + len("<think>"):].strip():
        return -1
    return idx


def ensure_thinking_off_sentinel(
    prompt: str,
    *,
    family_name: str | None = None,
    model_name: str | None = None,
    tools_present: bool = False,
) -> str:
    """Finish the prompt-side no-thinking contract for families that need it.

    Some R1-style templates suppress thinking by omitting the opening
    ``<think>`` tag when ``enable_thinking=False``. MiniMax-M2.7 still treats
    that bare assistant prefix as permission to open a visible reasoning block
    on exact-answer prompts. The stable prompt contract is an explicit empty
    thought sentinel: ``<think>\n</think>\n\n``.

    Do not add it for tool requests: tool selection often needs the model's
    planning rail, and existing server paths suppress reasoning in the stream
    instead of suppressing tool decisions in the prompt. LFM2 is not an
    exception: its official template does not implement ``enable_thinking`` and
    its converted bundle explicitly forbids a synthetic ``<think>`` prefill.
    The registry therefore advertises native reasoning only for that family and
    rejects a requested thinking-off mode instead of fabricating one here.
    """

    fam = (family_name or "").lower()
    name = (model_name or "").lower()
    is_minimax_m3 = fam in {"minimax_m3", "minimax_m3_vl"} or "minimax-m3" in name
    is_step3p7 = fam == "step3p7" or "step-3.7" in name
    is_minimax = (
        not is_minimax_m3
        and (fam == "minimax" or "minimax" in name)
    )

    last_open = trailing_open_think_index(prompt)
    if last_open >= 0:
        after_open = prompt[last_open + len("<think>") :]
        if "</think>" not in after_open:
            # These official templates always open this rail and have no
            # thinking-off branch. Preserve the native prompt; the public API
            # rejects instruct mode instead of fabricating an empty thought.
            if is_step3p7 or fam in {"glm5_next", "glm5_next_text"}:
                return prompt
            # #199-2B: MiniMax tool requests keep the planning rail open so the
            # model can still select tools; compatible families may close an
            # already-open thought when their native contract supports it.
            if tools_present and is_minimax:
                return prompt
            return prompt[: last_open + len("<think>")] + "\n</think>\n\n"
        return prompt

    # No open <think> in the prompt: tool requests keep the native rail intact.
    if tools_present:
        return prompt

    needs_empty_think = is_minimax
    if not needs_empty_think:
        return prompt

    return prompt.rstrip() + "\n<think>\n</think>\n\n"
