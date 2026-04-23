# SPDX-License-Identifier: Apache-2.0
"""Kimi K2.6 VLM loader — re-export of ``jang_tools.load_jangtq_kimi_vlm``.

Matches ``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.1 and §1.4 module
layout. Load path::

    from vmlx_engine.loaders.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model
    from vmlx_engine.vlm.generate_vl import generate_vl

    model, processor = load_jangtq_kimi_vlm_model("/path/Kimi-K2.6-REAP-30-JANGTQ_1L")
    result = generate_vl(
        model, processor,
        prompt="Describe this image briefly.",
        image="/path/cat.jpg",
        max_new_tokens=60,
        prefill_step_size=32,  # CRITICAL for 191 GB MoE; larger blows
                               # Metal's command-buffer watchdog.
    )

The Kimi entry point installs the ``kimi_k25`` → ``kimi_vl`` remap in
``mlx_vlm.utils.MODEL_REMAPPING`` (idempotent), drops the VL wired_limit
to ~52 % of RAM (vs the 70 % text-loader default) so Jetsam has headroom
on a 275 GB machine under SSD contention, and installs a vision ⇄
language command-buffer split on ``model.get_input_embeddings`` so the
first cold VL forward pass doesn't trip Metal's ~60 s watchdog.

Note: ``vmlx_engine`` also installs the ``kimi_k25`` remap at package
import time (see ``vmlx_engine/__init__.py``), so sessions that go
through vMLX's scheduler stack work even without calling this loader
directly. This module exists for doc parity and for scripts that want
the Kimi-specific memory/buffer tuning without going through the full
vMLX engine.
"""

from __future__ import annotations

try:
    from jang_tools.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model
except ImportError as _ie:  # pragma: no cover
    raise ImportError(
        "vmlx_engine.loaders.load_jangtq_kimi_vlm requires `jang_tools ≥ "
        "the build shipping load_jangtq_kimi_vlm.py` in the active Python "
        "environment. vMLX's bundled Python ships a compatible version."
    ) from _ie

__all__ = ["load_jangtq_kimi_vlm_model"]
