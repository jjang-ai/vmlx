"""Loader for Zyphra ZAYA text bundles."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

from ..models.zaya import register_mlx_lm_zaya
from ..utils.quant_shape_inference import infer_quant_overrides_for_bundle

logger = logging.getLogger(__name__)


def load_zaya_model(model_path: str | Path, *, lazy: bool = False):
    """Load a ZAYA BF16/MXFP4/affine bundle with the local CCA runtime.

    JANGTQ/MXTQ ZAYA bundles should still route through ``load_jang_model`` so
    jang_tools can replace ``switch_mlp`` projections with TurboQuant modules.
    """
    path = Path(model_path)
    register_mlx_lm_zaya()

    cfg = json.loads((path / "config.json").read_text())
    try:
        cfg = infer_quant_overrides_for_bundle(path, cfg)
    except Exception as exc:
        logger.debug("ZAYA quant-shape inference skipped: %s", exc)

    model, loaded_cfg = load_model(
        path,
        model_config=cfg,
        lazy=lazy,
        strict=True,
    )
    tokenizer = load_tokenizer(path, eos_token_ids=loaded_cfg.get("eos_token_id"))
    if not hasattr(model, "config"):
        model.config = loaded_cfg
    if not lazy:
        mx.eval(model.parameters())
    logger.info(
        "ZAYA runtime loaded: layers=%s, cache=CCA(KV+conv_state+prev_hs), "
        "prefix/paged/L2 disabled until restore tests pass",
        len(getattr(model, "layers", [])),
    )
    return model, tokenizer
