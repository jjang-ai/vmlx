# SPDX-License-Identifier: Apache-2.0
"""Chunked VL prefill helper — re-export of
``jang_tools.kimi_prune.generate_vl.generate_vl``.

Matches the module path prescribed in
``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.4 so doc readers can copy
the example verbatim::

    from vmlx_engine.loaders.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model
    from vmlx_engine.vlm.generate_vl import generate_vl

    model, processor = load_jangtq_kimi_vlm_model(path)
    result = generate_vl(
        model, processor,
        prompt="Describe this image briefly.",
        image="/path/cat.jpg",
        max_new_tokens=60,
        prefill_step_size=32,
    )
    print(result["text"], result["tok_per_sec"])

Why a re-export rather than a copy: the chunking contract is expressed
in a single place (``jang_tools.kimi_prune.generate_vl``) so the Swift
``ChunkedPrefillVLM.swift`` helper and this Python helper can't drift
apart silently.

DO NOT call ``mlx_vlm.generate`` directly on a JANGTQ_1L VLM bundle —
with ~100 image tokens that function skips chunked prefill entirely and
runs one monolithic Metal command buffer, exceeding the ~60 s GPU
watchdog on a 191 GB MoE (SIGABRT 134 /
``kIOGPUCommandBufferCallbackErrorTimeout``). Use this helper instead.
"""

from __future__ import annotations

try:
    from jang_tools.kimi_prune.generate_vl import generate_vl
except ImportError as _ie:  # pragma: no cover
    raise ImportError(
        "vmlx_engine.vlm.generate_vl requires `jang_tools.kimi_prune` in "
        "the active Python environment. vMLX's bundled Python ships it "
        "by default."
    ) from _ie

__all__ = ["generate_vl"]
