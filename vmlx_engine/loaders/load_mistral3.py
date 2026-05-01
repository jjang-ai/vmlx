# SPDX-License-Identifier: Apache-2.0
"""
Mistral 3.5 (Mistral-Medium-3.5-128B) loader — thin wrapper over
`jang_tools.mistral3.runtime.load`.

`model_type=mistral3` is the OUTER VLM wrapper class
(`Mistral3ForConditionalGeneration`). The INNER text model carries
`text_config.model_type=ministral3` — Mistral's renamed dense GQA
(96/8 heads, 128 head_dim, 88 layers, hidden 12288, 256K YaRN context).
The vision tower is `pixtral` (48 layers, hidden 1664, image_size 1540).

mlx_lm has no native `mistral3` or `ministral3` class, so the generic
loader falls back to `mistral` (the legacy 7B/8x22B class) which has
the wrong attention shapes and silently drops weights → garbage output.

Route to `jang_tools.mistral3.runtime.load` which auto-detects bundle
format (bf16 / JANG affine / JANGTQ / MXFP4) and instantiates the
right model class.

Bundle layout (per `~/jang/jang-tools/jang_tools/mistral3/README.md`):
- bf16 source: `model_type=mistral3`, `text_config.model_type=ministral3`
- JANGTQ2: `weight_format=mxtq` + `mxtq_bits` block; vision/projector/
  lm_head stay bf16 per `modules_to_not_convert`
- MXFP4:    `weight_format=mxfp4`, text decoder bits=4 group=32
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger("vmlx_engine.loaders.mistral3")


def load_mistral3_model(model_path: str | Path) -> Tuple[Any, Any]:
    """Load a Mistral-Medium-3.5 bundle and return `(model, tokenizer)`.

    Args:
        model_path: Bundle directory containing `config.json`,
            `tokenizer.json`, and packed safetensors shards.

    Returns:
        `(model, tokenizer)` tuple. `model.config` carries the raw
        `config.json` dict.

    Raises:
        ImportError: if `jang_tools.mistral3` is missing. Surface to
            the user as "install jang-tools >= the release that ships
            mistral3/".
    """
    try:
        from jang_tools.mistral3.runtime import load as _m3_load
    except ImportError as e:
        raise ImportError(
            "Mistral-Medium-3.5 bundle detected (model_type=mistral3 + "
            "text_config.model_type=ministral3) but `jang_tools.mistral3` "
            "is missing. Install with `pip install -U jang-tools>=2.5.0` "
            "(must include the mistral3/ submodule). Original error: " + str(e)
        )

    path = Path(model_path)
    logger.info("Loading Mistral-Medium-3.5 bundle: %s", path.name)

    model, cfg, fmt = _m3_load(str(path))
    logger.info(
        "Mistral-Medium-3.5 loaded: format=%s, text-layers=%d, "
        "vision-layers=%d, vocab=%d",
        fmt,
        cfg.text_config.num_hidden_layers,
        cfg.vision_config.num_hidden_layers,
        cfg.text_config.vocab_size,
    )

    try:
        cfg_path = path / "config.json"
        if cfg_path.is_file():
            model.config = json.loads(cfg_path.read_text())
    except Exception as _cfg_err:
        logger.debug("Mistral-Medium-3.5 model.config attach failed: %s", _cfg_err)
        model.config = {"model_type": "mistral3"}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True
    )

    return model, tokenizer
