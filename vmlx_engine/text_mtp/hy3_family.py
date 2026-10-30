# SPDX-License-Identifier: Apache-2.0
"""Head-attach gate for Tencent Hy3 (``hy_v3``).

All Hy3 model code, including the MTP head (``attach_mtp``, ``mtp_forward``,
``make_mtp_cache``, the ``return_hidden`` forward and an MTP-aware sanitize),
lives in ``jang_tools.hy3.model``, which registers itself as
``mlx_lm.models.hy_v3``. The head is a DeepSeek-V3-style next-token layer:
``eh_proj([enorm(embed(next)), hnorm(hidden)])`` -> one Hy3 MoE decoder layer
-> ``final_layernorm`` -> the shared lm_head; bundles ship it as ``mtp.0.*``.

The only vMLX piece is the gate: the head is attached at construction when
draft-head loading is on and the config declares ``num_nextn_predict_layers``.
A missing or old ``jang_tools`` is not an error: non-Hy3 sessions never need it.
"""

from __future__ import annotations

import logging
import sys

from ._marks import is_marked, mark

logger = logging.getLogger(__name__)


def install() -> bool:
    try:
        from jang_tools.hy3 import register_mlx_lm_hy3

        register_mlx_lm_hy3()
    except Exception as exc:
        logger.debug("hy_v3 head gate skipped (jang_tools.hy3 unavailable: %s)", exc)
        return True

    module = sys.modules.get("mlx_lm.models.hy_v3")
    model_cls = getattr(module, "Model", None) if module is not None else None
    if model_cls is None:
        logger.debug("hy_v3 head gate skipped (mlx_lm.models.hy_v3 not registered)")
        return True
    if not hasattr(model_cls, "attach_mtp"):
        logger.debug(
            "hy_v3 head gate skipped (jang_tools.hy3.model has no MTP head; "
            "upgrade jang for Hy3 speculative decode)"
        )
        return True
    if is_marked(model_cls):
        return True

    from . import draft_head_loading_enabled

    base_init = model_cls.__init__

    def __init__(self, args):
        base_init(self, args)
        declared = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        if declared > 0 and draft_head_loading_enabled():
            self.attach_mtp()

    model_cls.__init__ = __init__
    mark(model_cls)
    logger.info("hy_v3 native-MTP attach gate applied")
    return True
