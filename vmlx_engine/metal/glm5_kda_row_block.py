# SPDX-License-Identifier: Apache-2.0
"""Experimental value-row blocking of the incumbent GLM KDA AR operation.

Reuses q/k/g/beta across four values, retaining [B,H,K,V] FP32 state and the
incumbent per-value arithmetic (not a transposed recurrent-prefill algorithm).
No projection/quantization/normalization/cache-format change.
Default OFF until full-model and sustained serving qualification.
"""
from functools import lru_cache
import logging
import os

import mlx.core as mx

_LOG = logging.getLogger(__name__)
_CALLS = 0
MATH_ABI = "glm5_kda_rows4_incumbent_kv_v1"


def requested():
    return os.environ.get("VMLX_GLM5_KDA_ROW_BLOCK", "0") == "1"


@lru_cache(maxsize=1)
def _hardware_allowed():
    return (mx.__version__ in ("0.32.2", "0.32.3")
            and mx.metal.is_available()
            and mx.metal.device_info().get("architecture") == "applegpu_g17s")

@lru_cache(maxsize=1)
def _row_kernel(heads=64, width=128, rows=4):
    chunks = (width + 31) // 32
    source = f"""
        uint group = thread_position_in_grid.x / 32u;
        uint lane = thread_index_in_simdgroup;
        uint head = group / {width // rows}u;
        uint value0 = (group % {width // rows}u) * {rows}u;
        if (head >= {heads}u) return;
        float decayed[{rows}][{chunks}];
        float keys[{chunks}];
        float sums[{rows}];
        for (uint r=0; r<{rows}; ++r) sums[r] = 0.0f;
        for (uint chunk=0; chunk<{chunks}; ++chunk) {{
            uint ki = chunk*32u+lane;
            size_t ko = (size_t)head*{width}u+ki;
            float key_value = (float)key[ko];
            float decay = metal::exp((float)gate[ko]);
            keys[chunk] = key_value;
            for (uint r=0; r<{rows}; ++r) {{
                size_t so = ko*{width}u+value0+r;
                float state_value = (float)state[so]*decay;
                sums[r] += key_value*state_value;
                decayed[r][chunk] = state_value;
            }}
        }}
        float correction[{rows}];
        float out[{rows}];
        float bb = (float)beta[head];
        for (uint r=0; r<{rows}; ++r) {{
            correction[r] = (float)value[(size_t)head*{width}u+value0+r]
                - simd_sum(sums[r]);
            out[r] = 0.0f;
        }}
        for (uint chunk=0; chunk<{chunks}; ++chunk) {{
            uint ki=chunk*32u+lane;
            size_t ko=(size_t)head*{width}u+ki;
            for (uint r=0; r<{rows}; ++r) {{
                size_t so=ko*{width}u+value0+r;
                float next=decayed[r][chunk]+bb*keys[chunk]*correction[r];
                next_state[so]=next;
                out[r]+=(float)query[ko]*{width ** -0.5:.12f}f*next;
            }}
        }}
        for (uint r=0; r<{rows}; ++r) {{
            float y=simd_sum(out[r]);
            if(lane==0u) output[(size_t)head*{width}u+value0+r]=y;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_rows_h{heads}_d{width}_r{rows}",
        input_names=["query","key","value","gate","beta","state"],
        output_names=["output","next_state"],
        header="#include <metal_stdlib>\nusing namespace metal;\n", source=source,
    )


def glm5_kda_row_block(q, k, v, g, beta, state, *, enabled=None):
    """Return output/state or decline BEFORE submitting any mutation."""
    if enabled is None:
        enabled = requested()
    if not enabled or mx.default_device() != mx.gpu or not _hardware_allowed():
        return None
    if any(tuple(a.shape) != (1, 64, 128) for a in (q, k, v, g)):
        return None
    if tuple(beta.shape) != (1, 64) or tuple(state.shape) != (1, 64, 128, 128):
        return None
    if state.dtype != mx.float32:
        return None
    if any(a.dtype not in (mx.float16, mx.bfloat16, mx.float32)
           for a in (q, k, v, g, beta)):
        return None
    result = _row_kernel()(
        inputs=[q, k, v, g, beta, state],
        grid=(32 * 64 * 32, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 64, 128), tuple(state.shape)],
        output_dtypes=[mx.float32, mx.float32],
    )
    global _CALLS
    if not _CALLS:
        _LOG.info("GLM KDA row-block dispatch: rows=4 heads=64 key=128 value=128 "
                  "state=FP32 math=%s", MATH_ABI)
    _CALLS += 1
    return result


def observed_calls():
    return _CALLS

