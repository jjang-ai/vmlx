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
from typing import Any, Tuple

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


_OFF_VALUES = frozenset({"0", "false", "off", "no"})
_ON_VALUES = frozenset({"1", "true", "on", "yes"})


def dsv41_dspark_decision() -> Tuple[bool, str]:
    """Whether a V4.1 bundle's dSpark draft is attached at load, and the reason (logged as the receipt).

    Precedence (highest first):
      1. ``VMLX_DSPARK`` = the serve flag ``--dspark on|off``, which cli.serve exports as ``1`` / ``0`` before the model
         loads. It is the switch every DSpark family reads (dspark/runtime.load_dspark_runtime), so the app's DSpark
         checkbox and ``vmlx serve --dspark`` govern V4.1 like MiniCPM-V-4.7 and MiniMax-M3.1. An explicit choice is
         the user's: ``on`` attaches the draft even when the diagnostic lever below says 0.
      2. ``VMLX_DSV41_DSPARK=0`` (diagnostic lever, V4.1 only): plain AR when no explicit ``--dspark`` was given.
      3. ``--dspark auto`` (the default): attach whenever the bundle ships the draft (``runtime.bundle_has_mtp``).
    Draft depth is never chosen here: the generator's Adaptive governor picks every round's width."""
    import os

    choice = os.environ.get("VMLX_DSPARK", "").strip().lower()
    if choice in _OFF_VALUES:
        return False, "--dspark off (VMLX_DSPARK=0)"
    if choice in _ON_VALUES:
        return True, "--dspark on (VMLX_DSPARK=1)"
    lever = os.environ.get("VMLX_DSV41_DSPARK", "").strip().lower()
    if lever in _OFF_VALUES:
        return False, "VMLX_DSV41_DSPARK=0 (diagnostic lever; --dspark auto)"
    return True, "--dspark auto (bundle draft is used when shipped)"


def _maybe_load_dsv41_dspark(model, bundle: Path) -> None:
    """Load the bundle's self-contained dSpark MTP draft and attach it as ``model._dspark``; the scheduler then serves
    with DSV41DSparkGenerator (speculative decode, Adaptive width). Without ``model._dspark`` it serves the plain
    SingleBatchGenerator (AR). On by default for bundles that ship the draft; ``--dspark off`` (or the diagnostic
    ``VMLX_DSV41_DSPARK=0`` under ``--dspark auto``) serves AR. See ``dsv41_dspark_decision`` for the precedence.
    Loading is additive (the draft is a separate attached module) and the generator still falls back to AR
    per-request for VL turns."""
    import json

    wanted, reason = dsv41_dspark_decision()
    if not wanted:
        logger.info("DSV4.1: dSpark draft not attached: %s; serving AR (SingleBatchGenerator)", reason)
        return
    try:
        cfg = json.loads((bundle / "config.json").read_text())
    except Exception as exc:  # pragma: no cover
        logger.warning("DSV4.1: cannot read config.json for dSpark gate (%s)", exc)
        return
    rt = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
    if not rt.get("bundle_has_mtp"):
        logger.info("DSV4.1: bundle advertises no dSpark draft (runtime.bundle_has_mtp); serving AR (%s)", reason)
        return
    try:
        from jang_tools.dsv41.dspark import load_dspark

        runtime = getattr(model, "runtime", None)
        if runtime is None:
            logger.warning("DSV4.1: model wrapper exposes no .runtime; cannot attach dSpark draft")
            return
        ds = load_dspark(str(bundle), runtime.args, runtime.model.embed, runtime.model.head)
        model._dspark = ds
        logger.info("DSV4.1: dSpark MTP draft attached (model._dspark), bs=%s; %s", getattr(ds, "bs", "?"), reason)
    except Exception as exc:  # pragma: no cover
        logger.warning("DSV4.1: dSpark draft load failed (%s); serving AR-only", exc)


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
    # tool_choice reaches the template kwargs as for every model; the bundle encoder has no tool_choice input, so
    # nothing is rendered for it and the server enforces "required" after generation, as it does for other models.
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True, reasoning_effort=None, tools=None, **_):
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
