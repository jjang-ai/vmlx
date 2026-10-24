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


def _attach_chat_encoder(tokenizer, bundle: Path) -> None:
    """V4.1 ships a custom prompt encoder (encoding/encoding.py), not an HF chat_template. Attach an
    ``apply_chat_template`` shim that delegates to its ``encode_messages`` so the server's chat path
    (which calls ``renderer.apply_chat_template(messages, enable_thinking=..., tokenize=False)``)
    renders the native thinking/tool format."""
    enc_path = bundle / "encoding" / "encoding.py"
    if not enc_path.exists():
        logger.warning("DSV4.1: no encoding/encoding.py in bundle; chat endpoint will need a template")
        return
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("dsv41_bundle_encoding", enc_path)
        enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enc)
    except Exception as exc:  # pragma: no cover
        logger.warning("DSV4.1: failed to load bundle encoding (%s); chat endpoint unavailable", exc)
        return

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True, reasoning_effort="high", **_):
        mode = "thinking" if enable_thinking else "chat"
        prompt = enc.encode_messages(list(messages), thinking_mode=mode, reasoning_effort=reasoning_effort)
        if tokenize:
            try:
                return tokenizer.encode(prompt)
            except Exception:
                return tokenizer.encode(prompt).ids
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
