# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 native Python chat tokenizer loader (tokenization_ernie.py / encoding_ernie.py / decoding_ernie.py)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def load_ernie_tokenizer(model_path: str | Path) -> Any:
    """Load the bundle's native ErnieChatTokenizer locally (no Jinja, no network)."""
    root = Path(model_path)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    tokenizer_cls = get_class_from_dynamic_module(
        "tokenization_ernie.ErnieChatTokenizer",
        str(root),
        local_files_only=True,
    )
    tok = tokenizer_cls.from_pretrained(str(root), local_files_only=True)
    # Attach convenient helper for stop token IDs and reasoning effort normalization
    tok._vmlx_ernie_native = True
    return tok


def render_ernie_prompt_ids(
    tokenizer: Any,
    messages: List[dict],
    *,
    tools: Optional[List[dict]] = None,
    enable_thinking: bool = True,
    reasoning_effort: Optional[str] = "high",
    add_generation_prompt: bool = True,
) -> List[int]:
    """Render ERNIE 5.5 messages to exact token IDs using the bundle's native encoder."""
    mod = sys.modules.get(type(tokenizer).__module__)
    render_chat = getattr(mod, "render_chat", None)
    opts: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": bool(enable_thinking),
    }
    if enable_thinking and reasoning_effort:
        eff = str(reasoning_effort).strip().lower()
        if eff in ("low", "medium", "high"):
            opts["reasoning_effort"] = eff
        elif eff in ("xhigh", "max"):
            opts["reasoning_effort"] = "high"
        elif eff in ("minimal",):
            opts["reasoning_effort"] = "low"
        else:
            opts["reasoning_effort"] = "high"
    if tools:
        opts["tools"] = tools
    if callable(render_chat):
        rendered = render_chat(messages, **opts)
        return list(tokenizer.encode_segments(rendered.segments))
    return list(tokenizer.apply_chat_template(messages, tokenize=True, **opts))
