# SPDX-License-Identifier: Apache-2.0
"""One class marker for every hook this package installs.

The marker is looked up in the class's own ``__dict__``: mlx-lm subclasses
hooked classes (``qwen3_5_moe.Model`` extends ``qwen3_5.Model``), and an
inherited marker must not make the subclass look already hooked.
"""

from __future__ import annotations

from typing import Any

MARK = "__vmlx_text_mtp__"


def is_marked(cls: Any) -> bool:
    return MARK in vars(cls)


def mark(cls: Any, how: Any = True) -> None:
    """Record that ``cls`` is hooked; ``how`` says by whom ("patch"/"upstream")."""
    setattr(cls, MARK, how)


def marked_as(cls: Any) -> Any:
    return vars(cls).get(MARK)
