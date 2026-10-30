# SPDX-License-Identifier: Apache-2.0
"""Native multi-token prediction for vMLX's text lane.

A model that ships a next-token prediction head ("MTP head") can propose a few
tokens ahead of the backbone; one backbone forward over the proposals then
confirms the longest correct prefix. This package wires that into the mlx-lm
objects vMLX's text scheduler already drives:

* family hooks (``qwen_family``, ``hy3_family``) give the model a ``mtp`` head,
  ``mtp_forward``, ``make_mtp_cache`` and a ``__call__`` that can also return
  the pre-norm hidden state and split a verify forward at the confirmed row;
* ``hybrid_state`` lets recurrent (GatedDeltaNet) caches restore the state of
  an accepted prefix after a partial rejection;
* ``hooks`` wraps ``mlx_lm.generate.GenerationBatch`` so a singleton batch runs
  the draft/verify lane (``lane``) instead of one backbone step per token. The
  scheduler's caches (prefix, paged, SSD, TurboQuant) see only the confirmed
  tokens, because every rejected row is rewound before the next step.

Heads are attached at model construction time only while
``set_draft_head_loading(True)`` is in effect (``native_mtp`` decides that per
bundle), so a model loaded with native MTP off is identical to a stock build.
Batches with more than one live sequence always take the stock step.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_installed = False
_load_heads = False


def set_draft_head_loading(enabled: bool) -> None:
    """Choose whether models constructed from now on attach their MTP head.

    Loads run one at a time on the MLX executor, so a process-wide switch is
    enough. Model constructors read it through ``draft_head_loading_enabled``.
    """
    global _load_heads
    _load_heads = bool(enabled)


def draft_head_loading_enabled() -> bool:
    return _load_heads


def install() -> bool:
    """Install the family hooks and the generation-lane hooks (idempotent).

    Must run before ``mlx_lm.load()`` so the hooked constructors, ``from_dict``
    and ``sanitize`` see the head weights. Returns True when the lane is usable
    (now or from an earlier call); False when a required piece is missing.
    """
    global _installed
    if _installed:
        return True

    from . import hooks, hy3_family, hybrid_state, qwen_family

    if not hybrid_state.install_rollback_slot():
        return False

    for label, family in (("Qwen3.5/3.6", qwen_family), ("hy_v3", hy3_family)):
        try:
            ready = family.install()
        except Exception as exc:  # one family must not block the others
            ready = False
            logger.debug("%s MTP hooks raised: %s", label, exc)
        if not ready:
            logger.debug("%s MTP hooks not installed", label)

    if not hooks.install_generation_hooks():
        logger.warning(
            "BatchGenerator MTP dispatch patch failed; MTP path will be inactive"
        )
        return False

    _installed = True
    logger.info("mlx-lm MTP adapters applied")
    return True
