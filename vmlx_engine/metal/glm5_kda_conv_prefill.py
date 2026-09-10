"""Experimental exact-order KDA convolution sum; not enabled in model dispatch."""
from functools import lru_cache
import mlx.core as mx

@lru_cache(maxsize=8)
def _kernel(width):
    return mx.fast.metal_kernel(
        name=f"vmlx_kda_prefill_sum_w{width}",
        input_names=["x", "state", "weight", "dims"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=f"""
            uint c = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            uint C = dims[0], Tn = dims[1];
            if (c >= C || t >= Tn) return;
            float acc = 0.0f;
            for (uint w = 0; w < {width}u; ++w) {{
                uint pos = t + w;
                float value = pos < {width-1}u
                    ? (float)state[pos*C+c]
                    : (float)x[(pos-{width-1}u)*C+c];
                // Preserve separate FP32 multiply/add rounding of stock MLX.
                volatile float product = value * (float)weight[c*{width}u+w];
                acc = acc + product;
            }}
            out[t*C+c] = acc;
        """,
    )

def kda_conv_prefill(x, weight, state=None):
    """Return stock-shaped SiLU output and untouched-input trailing history.

    Unsupported geometry returns None before dispatch. This experimental
    component does not alter the production model or decode path.
    """
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] <= 1:
        return None
    if weight.ndim == 3:
        weight = weight.reshape(weight.shape[0], -1)
    if weight.ndim != 2 or weight.shape[0] != x.shape[2]:
        return None
    channels, width = map(int, weight.shape)
    if width < 2 or width > 8:
        return None
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return None
    if state is None:
        state = mx.zeros((1, width-1, channels), dtype=x.dtype)
    elif state.shape != (1, width-1, channels):
        return None
    state = state.astype(x.dtype)
    y, = _kernel(width)(
        inputs=[x, state, weight, mx.array([channels, x.shape[1]], dtype=mx.uint32)],
        grid=(channels, x.shape[1], 1),
        threadgroup=(min(128, channels), 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )
    y = (y * mx.sigmoid(y)).astype(x.dtype)
    if x.shape[1] >= width-1:
        tail = x[:, -(width-1):]
    else:
        tail = mx.concatenate([state, x], axis=1)[:, -(width-1):]
    return y, tail
