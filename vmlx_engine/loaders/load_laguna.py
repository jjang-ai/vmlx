# SPDX-License-Identifier: Apache-2.0
"""
Laguna (poolside) loader — thin wrapper over `jang_tools.laguna.runtime`.

Laguna is poolside.ai's 33B/3B agentic-coding MoE: 40 layers, hybrid
SWA + full attention with PER-LAYER head count (48 full / 64 SWA, dual
RoPE — full uses YaRN, SWA uses default), 256 routed experts top-8 + 1
shared, sigmoid routing with per-head gating (`g_proj`), q_norm/k_norm
in attention, dense layer 0 + sparse layers 1..39. Text-only.

`model_type = "laguna"`. Bundle layout (see
`/Users/example/jang/jang-tools/jang_tools/laguna/README.md`):

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

    # Tokenizer: standard HF route. Laguna ships a Qwen2-flavored
    # tokenizer.json + tokenizer_config.json + chat_template.jinja so no
    # custom encoder is needed (unlike DSV4 which has its own
    # `dsv4_chat_encoder.py`).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True
    )

    return model, tokenizer
