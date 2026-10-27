"""DeepSeek-V4.1 bundle chat encoder: one loaded module per bundle and the reasoning-effort contract.

The bundle ships ``encoding/encoding.py``; its ``render_reasoning_effort`` accepts an int 1..100 or one of
``{"low": 50, "high": 75, "max": 100}`` (default "high") and asserts on anything else. Server requests speak
the wider OpenAI/Anthropic effort vocabulary, so every V4.1 render (the loader's ``apply_chat_template`` shim for
text turns, ``dsv41_vl_preprocess`` for image turns, and the server's request normalization) maps the effort
through :func:`reasoning_budget` first. The ladder is the Swift runtime's
(``DeepseekV4ChatEncoder.reasoningBudget``): minimal 25, low 50, medium 62, high 75, xhigh 88, max 100;
ints and numeric strings clamp to 1..100. ``none`` (and the other thinking-off spellings the request models
accept) means thinking OFF, not a budget.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REASONING_BUDGETS: dict[str, int] = {
    "minimal": 25,
    "low": 50,
    "medium": 62,
    "high": 75,
    "xhigh": 88,
    "max": 100,
}

# Same spellings as vmlx_engine.api.models._NO_REASONING_EFFORTS: the request models already turn a top-level
# one of these into enable_thinking=False; nested chat_template_kwargs values get the same treatment here.
THINKING_OFF_EFFORTS = frozenset({"none", "off", "false", "disabled", "disable", "0"})


def is_thinking_off_effort(effort: Any) -> bool:
    return isinstance(effort, str) and effort.strip().lower() in THINKING_OFF_EFFORTS


def reasoning_budget(effort: Any) -> Optional[int]:
    """The encoder's numeric budget (1..100) for a requested effort, or None.

    None means "no budget requested or not recognized": the encoder then uses its own default ("high", 75).
    Thinking-off spellings also return None; callers check :func:`is_thinking_off_effort` first."""
    if effort is None or isinstance(effort, bool):
        return None
    if isinstance(effort, int):
        return max(1, min(100, effort))
    if isinstance(effort, float):
        return max(1, min(100, int(round(effort)))) if math.isfinite(effort) else None
    if not isinstance(effort, str):
        return None
    raw = effort.strip().lower()
    if not raw or raw in THINKING_OFF_EFFORTS:
        return None
    try:
        return max(1, min(100, int(raw)))
    except ValueError:
        pass
    try:
        value = float(raw)
    except ValueError:
        return REASONING_BUDGETS.get(raw)
    return max(1, min(100, int(round(value)))) if math.isfinite(value) else None


def encoder_effort(effort: Any) -> Optional[int]:
    """Value for ``encode_messages(reasoning_effort=...)``: an int 1..100, or None for the encoder default.

    Never returns a value the encoder's assertion rejects. An unrecognized string is logged and dropped."""
    budget = reasoning_budget(effort)
    if budget is None and effort is not None and not is_thinking_off_effort(effort):
        if not (isinstance(effort, str) and not effort.strip()):
            logger.warning(
                "DSV4.1: reasoning_effort %r is not a known effort (%s, or 1-100); using the encoder default",
                effort, ", ".join(REASONING_BUDGETS),
            )
    return budget


_ENCODING_CACHE: dict[str, Any] = {}
_ENCODING_LOCK = threading.Lock()


def load_bundle_encoding(model_path: str | Path):
    """The bundle's ``encoding/encoding.py`` module, loaded once per bundle directory (process-wide).

    The module is stateless across requests; the loader's chat-template shim and the VL preprocess share it."""
    bundle = Path(model_path).expanduser()
    try:
        key = str(bundle.resolve())
    except OSError:
        key = str(bundle)
    mod = _ENCODING_CACHE.get(key)
    if mod is not None:
        return mod
    with _ENCODING_LOCK:
        mod = _ENCODING_CACHE.get(key)
        if mod is not None:
            return mod
        p = bundle / "encoding" / "encoding.py"
        if not p.exists():
            raise FileNotFoundError(f"deepseek_v41: no encoding module at {p}")
        spec = importlib.util.spec_from_file_location(f"dsv41_bundle_encoding_{abs(hash(key))}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _ENCODING_CACHE[key] = mod
        return mod
