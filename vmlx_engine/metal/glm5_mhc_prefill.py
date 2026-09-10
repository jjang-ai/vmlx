"""Opt-in H=4 prefill Sinkhorn fusion; stock nonlinear/GEMM stages stay outside.

This candidate preserves FP32 normalization boundaries and leaves decode and
verification rows unchanged. Full-model performance qualification is separate.
"""
from functools import lru_cache
import logging
import os

import mlx.core as mx

_LOG = logging.getLogger(__name__)
_OBSERVED = False
_SOURCE = r"""
uint token = thread_position_in_grid.x;
if (token >= ROWS) return;
float a[16];
for (uint i = 0; i < 16; ++i) a[i] = comb[token * 16 + i];
float eps = epsilon[0];
for (uint iteration = 0; iteration < ITERS; ++iteration) {
    if (iteration > 0) {
        for (uint row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (uint col = 0; col < 4; ++col) sum += a[row * 4 + col];
            float denominator = sum + eps;
            for (uint col = 0; col < 4; ++col)
                a[row * 4 + col] = a[row * 4 + col] / denominator;
        }
    }
    for (uint col = 0; col < 4; ++col) {
        float sum = 0.0f;
        for (uint row = 0; row < 4; ++row) sum += a[row * 4 + col];
        float denominator = sum + eps;
        for (uint row = 0; row < 4; ++row)
            a[row * 4 + col] = a[row * 4 + col] / denominator;
    }
}
for (uint i = 0; i < 16; ++i) out[token * 16 + i] = a[i];
"""


def fused_glm5_mhc_prefill_requested() -> bool:
    return os.environ.get("VMLX_GLM5_FUSED_MHC_PREFILL", "0").lower().strip() in {
        "1", "true", "yes", "on"
    }


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_prefill_sinkhorn_h4",
        input_names=["comb", "epsilon"], output_names=["out"],
        source=_SOURCE, compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=16)
def _epsilon(value):
    return mx.array([value], dtype=mx.float32)


def glm5_mhc_prefill_sinkhorn(comb, *, sink_eps, iterations, enabled=None):
    """Return None for disabled/unsupported layouts so the caller stays stock."""
    if enabled is None:
        enabled = fused_glm5_mhc_prefill_requested()
    if not enabled or comb.ndim != 4 or comb.dtype != mx.float32:
        return None
    batch, rows, height, width = comb.shape
    if batch != 1 or rows <= 4 or (height, width) != (4, 4):
        return None
    if not isinstance(iterations, int) or not 1 <= iterations <= 64:
        return None
    result = _kernel()(
        inputs=[comb, _epsilon(float(sink_eps))],
        template=[("ROWS", rows), ("ITERS", iterations)],
        grid=(rows, 1, 1), threadgroup=(min(256, rows), 1, 1),
        output_shapes=[comb.shape], output_dtypes=[mx.float32],
    )[0]
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
        _LOG.info("mHC prefill Sinkhorn candidate dispatched: shape=%s iterations=%s", comb.shape, iterations)
    return result
