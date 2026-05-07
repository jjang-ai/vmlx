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


def build_chat_template_kwargs(
    *,
    enable_thinking: bool | None,
    extra: Mapping[str, Any] | None = None,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    include_thinking_alias: bool = True,
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

    return kwargs
