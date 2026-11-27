# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned native MTP adapters for mlx-vlm model copies."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_mlx_vlm_mtp_patch() -> bool:
    """Install native MTP runtime methods on supported mlx-vlm model copies.

    Delegates idempotency to ``qwen35_vl.apply()`` (which tracks patched module
    identities in ``_PATCHED_MODULES``) so registering the vendored
    ``qwen3_5_family`` runtime after an earlier patch still patches the new
    module objects.
    """
    global _PATCHED
    from . import qwen35_vl

    qwen_ok = qwen35_vl.apply()
    if not qwen_ok:
        logger.warning("mlx-vlm Qwen3.5/3.6 MTP adapter failed to apply")
        return False

    _patch_load_config_mtp_quant_aliases()

    if not _PATCHED:
        _PATCHED = True
        logger.info("mlx-vlm native MTP adapters applied")
    return True


def _patch_load_config_mtp_quant_aliases() -> None:
    """Mirror `mtp.*` quantization overrides to `language_model.mtp.*`.

    v3 capability-schema bundles (Qwen3.6-27B D-series) stamp the MTP head's
    per-module quantization entries under the head's OWN key space —
    ``mtp.fc``, ``mtp.layers.0.self_attn.q_proj`` — while the module our
    adapter constructs lives at ``language_model.mtp.*``. mlx_vlm's
    ``get_class_predicate`` does an exact ``p in config["quantization"]``
    lookup, so every head module missed its override and fell back to the
    top-level bits: the head is stamped 6-bit, the fallback packed for 4-bit,
    and load_weights failed with
    ``Expected shape (5120, 1280) but received shape (5120, 1920) for
    parameter language_model.mtp.fc.weight``.

    The predicate is a closure inside ``load_model``, so the seam is the
    config it reads: alias the keys in ``load_config``'s result. Additive
    only — existing keys are never overwritten, and configs without ``mtp.*``
    entries pass through untouched.
    """
    try:
        from mlx_vlm import utils as _vlm_utils
    except Exception as e:  # pragma: no cover - mlx_vlm absent in some CI venvs
        logger.debug(f"mtp quant alias patch skipped: {e}")
        return
    if getattr(_vlm_utils.load_config, "_vmlx_mtp_quant_aliased", False):
        return

    _original_load_config = _vlm_utils.load_config

    def load_config_with_mtp_aliases(model_path, **kwargs):
        config = _original_load_config(model_path, **kwargs)
        try:
            quant = config.get("quantization")
            if isinstance(quant, dict):
                aliases = {
                    f"language_model.{key}": value
                    for key, value in quant.items()
                    if isinstance(value, dict)
                    and key.startswith("mtp.")
                    and f"language_model.{key}" not in quant
                }
                if aliases:
                    quant.update(aliases)
        except Exception as alias_err:  # never break loading over an alias
            logger.debug(f"mtp quant alias skipped: {alias_err}")
        return config

    load_config_with_mtp_aliases._vmlx_mtp_quant_aliased = True
    _vlm_utils.load_config = load_config_with_mtp_aliases
    logger.info("mlx-vlm load_config patched: mtp.* quant overrides aliased")
