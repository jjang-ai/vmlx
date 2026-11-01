# SPDX-License-Identifier: Apache-2.0
"""Qualified GLM KDA pairwise row reduction without a product buffer.

The qualified MLX 0.32.2 width-128 contiguous sum uses 32 lanes, four
consecutive elements per lane, initial +0, product + accumulator, then
simd_sum. Preserve that tree and separate FP32 product rounding. This only
replaces bulk-prefill pairwise sums, not recurrence, chunking, or cache state.
Unsupported runtimes/shapes and first-launch failures retain the stock path.
Automatic use is M5 Max/MLX 0.32.2 or 0.32.3/B1; other existing opt-in shapes remain
experimental. Selection policy lives in glm5_prefill_policy and the caller.
"""

from functools import lru_cache
import importlib.metadata
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False

_SOURCE = r"""
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const size_t pair = (size_t)thread_position_in_grid.x / 32;
    const uint lane = thread_position_in_grid.x % 32;
    if (pair >= (size_t)ROWS) return;
    const uint column = uint(pair % 64);
    const uint row = uint((pair / 64) % 64);
    const size_t outer = pair / (64 * 64);
    const size_t li = (outer * 64 + row) * 128 + lane * 4;
    const size_t ri = (outer * 64 + column) * 128 + lane * 4;
    float total = 0.0f;
    for (uint d = 0; d < 4; ++d) {
        float delta = float(gates[li + d]) - float(gates[ri + d]);
        float bounded = metal::min(delta, 0.0f);
        float decay = metal::precise::exp(bounded);
        float first = float(left[li + d]) * decay;
        float product = first * float(right[ri + d]);
        total = product + total;
    }
    total = metal::simd_sum(total);
    if (lane == 0) sums[pair] = total;
"""


@lru_cache(maxsize=1)
def _compatible_runtime() -> bool:
    # A public sum API does not promise its reduction tree. Other MLX/GPU
    # versions remain on the existing path until separately qualified.
    try:
        version = importlib.metadata.version("mlx")
        device = mx.device_info().get("device_name", "")
        return (
            (version == "0.32.2" and "Apple M5" in device)
            or (version in ("0.32.3", "0.32.5.dev20261101+360078a8") and device == "Apple M5 Max")
        )
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_exact_register_pairwise_sum_v1",
        input_names=["left", "right", "gates"],
        output_names=["sums"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def glm5_pairwise_sum(left, right, gates, *, enabled: bool):
    """Return the qualified sum, or None for the existing product + mx.sum.

    All eligibility comes from actual operand shape/dtype and the installed
    reduction implementation, never a model-folder quantization label.
    No input, recurrent state, or output arrays are retained in Python.
    """
    global _FAILED, _OBSERVED
    if not enabled or _FAILED:
        return None
    if (
        left.ndim != 4
        or left.shape != right.shape
        or left.shape != gates.shape
        or tuple(left.shape[-2:]) != (64, 128)
        or left.size == 0
        or any(x.dtype != mx.float32 for x in (left, right, gates))
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or not _compatible_runtime()
    ):
        return None
    rows = int(left.shape[0]) * int(left.shape[1]) * 64 * 64
    # Metal's thread-position component is uint32 even though offsets use
    # size_t. Do not permit its flattened dispatch to wrap.
    if rows * 32 >= 2**32:
        return None
    try:
        result = _kernel()(
            inputs=[left, right, gates],
            template=[("ROWS", rows)],
            grid=(rows * 32, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(*left.shape[:-1], 64)],
            output_dtypes=[mx.float32],
        )[0]
        if not _OBSERVED:
            # Surface first compilation/launch failure before passing lazy
            # output into a native-state consumer. Subsequent calls stay lazy.
            mx.eval(result)
            _OBSERVED = True
            logger.info(
                "GLM exact register pairwise sum active: shape=%s dtype=fp32 "
                "scope=prefill reduction=mlx0322_row128_v1 product_buffer=false",
                tuple(left.shape),
            )
        return result
    except (RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning(
            "GLM register pairwise sum unavailable; retaining stock reduction: %s",
            exc,
        )
        return None
