# SPDX-License-Identifier: Apache-2.0
"""
Laguna (poolside) loader — thin wrapper over `jang_tools.laguna.runtime`.

Laguna is poolside.ai's 33B/3B agentic-coding MoE: 40 layers, hybrid
SWA + full attention with PER-LAYER head count (48 full / 64 SWA, dual
RoPE — full uses YaRN, SWA uses default), 256 routed experts top-8 + 1
shared, sigmoid routing with per-head gating (`g_proj`), q_norm/k_norm
in attention, dense layer 0 + sparse layers 1..39. Text-only.

`model_type = "laguna"`. Bundle layout (see
`/Users/eric/jang/jang-tools/jang_tools/laguna/README.md`):

  - bf16 reference   (`weight_format` absent or "bf16")
  - JANG affine      (`quantization.bits` set)
  - JANGTQ           (`weight_format == "mxtq"` + `mxtq_bits` block)
  - MXFP4            (`weight_format == "mxfp4"`)

`jang_tools.laguna.runtime.load(src)` auto-detects the format and
returns `(model, cfg, fmt)`. This wrapper adapts that to the
`(model, tokenizer)` contract vmlx_engine's loader chain expects, and
attaches `model.config` (raw `config.json` dict) so downstream code
that introspects model architecture works the same way it does for
DSV4 / Kimi / MiniMax JANGTQ bundles.

The chat template + tokenizer come from the bundle's standard
`tokenizer_config.json` + `tokenizer.json` via HuggingFace
`AutoTokenizer.from_pretrained` — Laguna ships a Qwen2-flavored
tokenizer (vocab=100352) so no custom encoder is needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger("vmlx_engine.loaders.laguna")


def load_laguna_model(model_path: str | Path) -> Tuple[Any, Any]:
    """Load a Laguna bundle and return `(model, tokenizer)`.

    Args:
        model_path: Path to the bundle directory containing `config.json`,
            `jang_config.json` (optional), `tokenizer.json`, and packed
            safetensors shards.

    Returns:
        A `(model, tokenizer)` tuple. `model.config` carries the raw
        `config.json` dict so downstream code can introspect.

    Raises:
        ImportError: if `jang_tools.laguna` is not installed. Surface this
            to the user as "install jang-tools >= the release that ships
            laguna/" — same convention as the DSV4 loader.
    """
    try:
        from jang_tools.laguna.runtime import load as _laguna_load
    except ImportError as e:
        raise ImportError(
            "Laguna bundle detected (model_type=laguna) but `jang_tools.laguna` "
            "is missing. Install with `pip install -U jang-tools>=2.5.0` "
            "(must include the laguna/ submodule). Original error: " + str(e)
        )

    path = Path(model_path)
    logger.info("Loading Laguna bundle: %s", path.name)

    # 2026-05-02 follow-up: Laguna JANGTQ now wired through
    # `jang_tools.jangrt.jangtq_hydrate.hydrate_jangtq` (jang-tools 2.5.12+).
    # The downstream `_laguna_load` swaps `.tq_packed`-bearing modules to
    # TurboQuant{Linear,SwitchLinear} before `model.update`. Older jang-tools
    # without the helper raises `ImportError` → caught here as a clean
    # NotImplementedError pointing the user at the alternative bundles.
    try:
        cfg_check = json.loads((path / "config.json").read_text())
        if cfg_check.get("weight_format") == "mxtq" or "mxtq_bits" in cfg_check:
            try:
                # Surface a clear error if the helper isn't installed.
                from jang_tools.jangrt.jangtq_hydrate import hydrate_jangtq  # noqa: F401
            except ImportError as _ie:
                raise NotImplementedError(
                    "Laguna JANGTQ (weight_format=mxtq) needs jang-tools "
                    f">= 2.5.12 (jangrt.jangtq_hydrate is missing: {_ie}). "
                    "Either upgrade jang-tools, or use the bf16 / "
                    "JANG-affine / MXFP4 variant of this bundle. "
                    "Path: " + str(path)
                ) from _ie
    except (OSError, json.JSONDecodeError):
        pass

    model, cfg, fmt = _laguna_load(str(path))
    logger.info(
        "Laguna loaded: format=%s, layers=%d, experts=%d, "
        "vocab=%d, hidden=%d",
        fmt, cfg.num_hidden_layers, cfg.num_experts,
        cfg.vocab_size, cfg.hidden_size,
    )

    # Attach the raw HF config dict to model.config for parity with the
    # mlx_lm.load() contract used elsewhere. Downstream code (cache config,
    # parser detection, registry lookups) reads model.config["model_type"]
    # and similar fields without caring whether the load went through
    # mlx_lm or our routed laguna runtime.
    try:
        cfg_path = path / "config.json"
        if cfg_path.is_file():
            model.config = json.loads(cfg_path.read_text())
    except Exception as _cfg_err:
        logger.debug("Laguna model.config attach failed: %s", _cfg_err)
        model.config = {"model_type": "laguna"}

    # Tokenizer: poolside-flavored. Laguna ships
    # tokenizer.json + tokenizer_config.json + chat_template.jinja with
    # `eos_token = '〈|EOS|〉'` (token id 2) — but the chat-template
    # closes assistant turns with `</assistant>\n` (token id 24, NOT a
    # special token but a single vocab entry). `generation_config.json`
    # encodes this correctly as `eos_token_id: [2, 24]`. Stock HF
    # `AutoTokenizer.from_pretrained` only picks up `eos_token_id` from
    # tokenizer_config, so the loaded tokenizer has
    # `eos_token_ids == 2` and `mlx_lm.stream_generate` stops only on
    # 2. The model was trained to emit 24 at the natural turn boundary,
    # so without 24 in the stop set, it emits `</assistant>\n`,
    # continues past it, and starts hallucinating a new assistant turn
    # (the symptom users see as "looping"). Read the full eos id list
    # from generation_config.json and attach it to the tokenizer so
    # mlx_lm honors both.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True
    )
    # mlx_lm.generate.stream_generate wraps the tokenizer with
    # `TokenizerWrapper(tokenizer)` — no `eos_token_ids` arg passed —
    # which falls back to `{tokenizer.eos_token_id}` (singleton). For
    # Laguna that's `{2}` (`〈|EOS|〉`) only, missing token 24
    # (`</assistant>`) which is the chat-template's natural assistant
    # turn close. The model is trained to emit token 24 at the end of
    # an answer; without it in the stop set the engine continues past
    # the boundary, emitting `</assistant>\n</assistant>\n...` and
    # hallucinating new turns (the user-visible "looping" symptom).
    # Pre-wrap with the full eos id list from generation_config.json
    # so `isinstance(tokenizer, TokenizerWrapper)` short-circuits the
    # re-wrap and our explicit eos_token_ids list survives.
    try:
        gen_cfg_path = path / "generation_config.json"
        eos_ids: list[int] = []
        if gen_cfg_path.is_file():
            gen_cfg = json.loads(gen_cfg_path.read_text())
            raw = gen_cfg.get("eos_token_id")
            if isinstance(raw, list):
                eos_ids = list(dict.fromkeys(int(t) for t in raw))
            elif isinstance(raw, int):
                eos_ids = [int(raw)]
        if eos_ids:
            from mlx_lm.tokenizer_utils import TokenizerWrapper
            tokenizer = TokenizerWrapper(tokenizer, eos_token_ids=eos_ids)
            logger.info(
                "Laguna tokenizer pre-wrapped with eos_token_ids=%s", eos_ids,
            )
    except Exception as _eos_err:
        logger.warning("Laguna eos_token_ids attach failed: %s", _eos_err)

    return model, tokenizer
