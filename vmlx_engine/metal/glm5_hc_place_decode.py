"""Single-dispatch GLM-5.3 hyper-connection placement for decode/verify."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def fused_glm5_hc_place_requested() -> bool:
    # default on since 2026-10-10: +1.3% GLM decode, KL 0.005 (inside the split-order floor 0.002-0.005)
    value = os.environ.get("VMLX_GLM5_FUSED_HC_PLACE", "1").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def fused_glm5_hc_place_verify_requested() -> bool:
    # Default on so native-MTP verification slabs (2-4 rows) stay fused when
    # the base hc_place path is enabled. ``VMLX_GLM5_FUSED_HC_PLACE_VERIFY=0``
    # is the explicit stock-path rollback; the VMLINUX-prefixed name is a
    # retained legacy alias.
    value = os.environ.get(
        "VMLINUX_GLM5_FUSED_HC_PLACE_VERIFY",
        os.environ.get("VMLX_GLM5_FUSED_HC_PLACE_VERIFY", "1"),
    ).strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _kernel(streams: int, hidden: int, rows: int):
    source = f"""
        uint index = thread_position_in_grid.x;
        if (index >= {rows * streams * hidden}u) return;
        uint token_target = index / {hidden}u;
        uint token = token_target / {streams}u;
        uint target = token_target % {streams}u;
        uint dim = index % {hidden}u;

        size_t stream_base = (size_t)token * {streams * hidden}u;
        size_t comb_base = (size_t)token * {streams * streams}u;
        float value = float(T(post[(size_t)token * {streams}u + target])) *
            float(block_out[(size_t)token * {hidden}u + dim]);
        for (uint source = 0u; source < {streams}u; ++source) {{
            value += float(T(comb[comb_base + source * {streams}u + target])) *
                float(residual[stream_base + source * {hidden}u + dim]);
        }}
        output[index] = T(value);
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_hc_place_h{streams}_d{hidden}",
        input_names=["post", "comb", "block_out", "residual"],
        output_names=["output"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def glm5_hc_place_decode(
    post: mx.array,
    comb: mx.array,
    block_out: mx.array,
    residual: mx.array,
    *,
    enabled: bool,
    verify_enabled: bool | None = None,
) -> mx.array | None:
    """Return placed streams for the exact single-token shape or ``None``."""

    if not enabled or residual.ndim != 4:
        return None
    batch, rows, streams, hidden = (int(value) for value in residual.shape)
    if batch != 1 or not 1 <= rows <= 4 or streams != 4 or hidden <= 0:
        return None
    if rows > 1:
        if verify_enabled is None:
            verify_enabled = fused_glm5_hc_place_verify_requested()
        if not verify_enabled:
            return None
    if tuple(post.shape) != (1, rows, streams):
        return None
    if tuple(comb.shape) != (1, rows, streams, streams):
        return None
    if tuple(block_out.shape) != (1, rows, hidden):
        return None
    supported = (mx.float16, mx.bfloat16, mx.float32)
    if residual.dtype not in supported or block_out.dtype != residual.dtype:
        return None
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return None

    output = _kernel(streams, hidden, rows)(
        inputs=[post, comb, block_out, residual],
        template=[("T", residual.dtype)],
        grid=(rows * streams * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(residual.shape)],
        output_dtypes=[residual.dtype],
    )[0]
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return output


def glm5_hc_place_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_glm5_hc_place_requested",
    "fused_glm5_hc_place_verify_requested",
    "glm5_hc_place_decode",
    "glm5_hc_place_status",
]
