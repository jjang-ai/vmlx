# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash (``model_type: deepseek_v41``) loader for the vMLX engine.

V4.1 bundles are JANG affine (per-tensor q2/q3) with a V4-family latent-attention backbone plus
Engram n-gram taps, a vision tower and dSpark MTP heads. The validated batch-1 runtime lives in
``jang_tools.dsv41``; ``jang_tools.dsv4_1.mlx_model.load_deepseek_v41`` builds it and wraps it in the
engine's Model contract (``make_cache`` / ``__call__(input_ids, cache, mask)``). This thin wrapper just
pairs that model with a standard engine tokenizer and keeps the DSV4.1 entry symmetric with
``load_jangtq_dsv4``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_jangtq_dsv41_model(model_path: str, *, skip_params_eval: bool = False) -> Tuple[Any, Any]:
    """Return ``(model, tokenizer)`` for a DeepSeek-V4.1-Flash JANG bundle."""
    from jang_tools.dsv4_1.mlx_model import load_deepseek_v41

    model, raw_tokenizer = load_deepseek_v41(model_path, skip_params_eval=skip_params_eval)
    _maybe_load_dsv41_dspark(model, Path(model_path))

    # A standard engine tokenizer (chat template / eos / detokenizer) over the bundle files.
    tokenizer = None
    for _loader in ("mlx_lm.utils", "mlx_lm.tokenizer_utils"):
        try:
            import importlib

            _mod = importlib.import_module(_loader)
            _fn = getattr(_mod, "load_tokenizer", None)
            if _fn is not None:
                tokenizer = _fn(Path(model_path))
                break
        except Exception as exc:  # pragma: no cover
            logger.debug("DSV4.1 tokenizer via %s failed: %s", _loader, exc)
    if tokenizer is None:
        logger.warning("DSV4.1: standard tokenizer load failed; using the bundle's raw tokenizer")
        tokenizer = raw_tokenizer

    _attach_chat_encoder(tokenizer, Path(model_path))
    return model, tokenizer


def _maybe_load_dsv41_dspark(model, bundle: Path) -> None:
    """Load the bundle's self-contained dSpark MTP draft and attach it as ``model._dspark`` so a V4.1
    spec-decode generator can drive it (DECODE_DSPARK_METHODS.md M1). DSpark adaptive is now the
    DEFAULT (per Eric 2026-10-25: 50+ tok/s easy code) — the generator + prefix-cache coexistence are
    proven; set ``VMLX_DSV41_DSPARK=0`` to force the plain AR path. Loading is additive (the draft is a
    separate attached module) and the generator still falls back to AR per-request for VL turns."""
    import json
    import os

    if os.environ.get("VMLX_DSV41_DSPARK", "1").strip().lower() in {"0", "false", "off", "no"}:
        return
    try:
        cfg = json.loads((bundle / "config.json").read_text())
    except Exception as exc:  # pragma: no cover
        logger.warning("DSV4.1: cannot read config.json for dSpark gate (%s)", exc)
        return
    rt = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
    if not rt.get("bundle_has_mtp"):
        logger.info("DSV4.1: VMLX_DSV41_DSPARK set but bundle advertises no MTP draft; serving AR-only")
        return
    try:
        from jang_tools.dsv41.dspark import load_dspark

        runtime = getattr(model, "runtime", None)
        if runtime is None:
            logger.warning("DSV4.1: model wrapper exposes no .runtime; cannot attach dSpark draft")
            return
        ds = load_dspark(str(bundle), runtime.args, runtime.model.embed, runtime.model.head)
        model._dspark = ds
        logger.info("DSV4.1: dSpark MTP draft attached (model._dspark), bs=%s", getattr(ds, "bs", "?"))
    except Exception as exc:  # pragma: no cover
        logger.warning("DSV4.1: dSpark draft load failed (%s); serving AR-only", exc)


# The same text as the Swift runtime's DeepseekV4ChatEncoder.requiredToolChoiceReminder for V4.1.
_REQUIRED_TOOL_REMINDER = (
    "The active API tool_choice is required for this assistant turn. "
    "Call exactly one available tool now using a <\uff5cDSML\uff5c calls> block. "
    "Do not answer in prose before the tool result."
)


def _required_tool_reminder(tool_choice, tools) -> Optional[str]:
    """The latest_reminder text for a required / named ``tool_choice`` with tools present, else None.

    The bundle encoder has no tool_choice parameter, and without a cue the model often answers in prose (the server
    then fails the request). Like the Swift runtime, the turn carries a ``latest_reminder`` naming the function when
    the choice names one or only one tool is available."""
    if not tools:
        return None
    if tool_choice == "required":
        name = None
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = ((tool_choice.get("function") or {}).get("name") or "").strip() or None
        if name is None:
            return None
    else:
        return None
    if name is None:
        names = {((t.get("function") or t).get("name") or "").strip() for t in tools if isinstance(t, dict)}
        names.discard("")
        name = next(iter(names)) if len(names) == 1 else None
    return _REQUIRED_TOOL_REMINDER + (f" Use the `{name}` function." if name else "")


def _with_reminder(msgs: list, content: str) -> list:
    """Insert a latest_reminder after the last user / developer / tool message (appended when there is none), unless
    the conversation already ends with one. The encoder merges tool results into user turns, so in a tool loop the
    reminder follows the latest results, where the Swift runtime (which inserts after merging) puts it too."""
    if msgs and msgs[-1].get("role") == "latest_reminder":
        return msgs
    tail = max((i for i, m in enumerate(msgs) if m.get("role") in ("user", "developer", "tool")), default=None)
    at = len(msgs) if tail is None else tail + 1
    return msgs[:at] + [{"role": "latest_reminder", "content": content}] + msgs[at:]


def _attach_chat_encoder(tokenizer, bundle: Path) -> None:
    """V4.1 ships a custom prompt encoder (encoding/encoding.py), not an HF chat_template. Attach an
    ``apply_chat_template`` shim that delegates to its ``encode_messages`` so the server's chat path
    (which calls ``renderer.apply_chat_template(messages, enable_thinking=..., tokenize=False)``)
    renders the native thinking/tool format. The module is the one ``dsv41_vl_preprocess`` uses for
    image turns (``chat_encoding.load_bundle_encoding``), so both paths render with the same code."""
    from ..models.deepseek_v4_1.chat_encoding import (
        encoder_effort,
        is_thinking_off_effort,
        load_bundle_encoding,
    )

    if not (bundle / "encoding" / "encoding.py").exists():
        logger.warning("DSV4.1: no encoding/encoding.py in bundle; chat endpoint will need a template")
        return
    try:
        enc = load_bundle_encoding(bundle)
    except Exception as exc:  # pragma: no cover
        logger.warning("DSV4.1: failed to load bundle encoding (%s); chat endpoint unavailable", exc)
        return

    # add_generation_prompt is accepted but not applied: encode_messages always ends a user/tool turn with
    # the assistant rail (`<｜Assistant｜><think>` or `</think>`). That keeps the engine's measured
    # generation-prompt length at 0 for V4.1, which is what its prefix cache needs: the stored state is the
    # exact N-1 prompt boundary (DeepseekV41Cache cannot be trimmed), and N-1 already drops the trailing
    # `<think>` / `</think>` token, so the next turn's history (`<｜Assistant｜></think>answer...`) extends it.
    # thinking_budget (from max_thinking_tokens / budget_tokens) is ignored: V4.1 steers reasoning depth
    # with the numeric reasoning effort only (the server logs that once per request).
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True, reasoning_effort=None, tools=None, tool_choice=None, **_):
        thinking_on = bool(enable_thinking) and not is_thinking_off_effort(reasoning_effort)
        mode = "thinking" if thinking_on else "chat"
        msgs = [dict(m) for m in messages]
        # V4.1 renders tools natively (render_tools) ONLY inside a system message
        # (encode_messages reads msg["tools"]). The server passes tools here as a
        # kwarg; attach them to a leading system turn so the native DSML tool
        # schema is rendered (otherwise the server injects a generic fallback
        # schema and the model emits a non-DSML call -> 422).
        if tools:
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["tools"] = tools
            else:
                msgs = [{"role": "system", "content": "", "tools": tools}] + msgs
        reminder = _required_tool_reminder(tool_choice, tools)
        if reminder:
            msgs = _with_reminder(msgs, reminder)
        prompt = enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=encoder_effort(reasoning_effort))
        if tokenize:
            ids = tokenizer.encode(prompt)  # mlx_lm wrapper -> list[int]; raw tokenizers.Tokenizer -> Encoding
            return list(getattr(ids, "ids", ids))
        return prompt

    # Attach to the wrapper and its inner HF tokenizer so the server's renderer discovery finds it.
    for target in (tokenizer, getattr(tokenizer, "_tokenizer", None), getattr(tokenizer, "tokenizer", None)):
        if target is not None:
            try:
                target.apply_chat_template = apply_chat_template
                # A non-empty marker so any HF-side "chat_template is set" guard passes.
                if getattr(target, "chat_template", None) in (None, ""):
                    target.chat_template = "{# dsv4.1 native encoder #}"
            except Exception:
                pass
    tokenizer._dsv41_encoding = enc  # keep a reference alive
