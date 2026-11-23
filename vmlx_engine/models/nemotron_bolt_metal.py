# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for the Nemotron Bolt LiLiCorr draft and verifier."""

from functools import cache
from importlib.metadata import PackageNotFoundError, version
import os

import mlx.core as mx


def available():
    if os.environ.get("VMLX_NEMOTRON_LILICORR_FUSED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        return (
            mx.metal.is_available()
            and mx.device_info().get("device_name") == "Apple M5 Max"
            and version("mlx") == "0.32.5.dev20261101+360078a8"
            and version("mlx-lm") == "0.31.3"
        )
    except (PackageNotFoundError, RuntimeError):
        return False


@cache
def _recurrent_kernel():
    return mx.fast.metal_kernel(
        name="bolt_mamba_ssm_sequence_shape",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_confirmed", "state_final"],
        source="""
            // Keep mlx-lm's single-row operation order and FP32 recurrent state.
            // Read the sequence length from the input to reuse the pipeline across draft widths.
            const uint h = thread_position_in_grid.z;
            const uint g = h / HeadsPerGroup;
            const uint d = thread_position_in_grid.y;
            const uint lane = thread_position_in_threadgroup.x;
            constexpr uint PerLane = StateSize / 32;
            const uint s0 = lane * PerLane;
            const uint base = (h * HeadDim + d) * StateSize + s0;
            const float A = -fast::exp(static_cast<float>(A_log[h]));
            float state[PerLane];
            for (uint i = 0; i < PerLane; ++i) {
                state[i] = static_cast<float>(state_in[base + i]);
            }
            for (uint row = 0; row < X_shape[1]; ++row) {
                const uint x_idx = (row * Heads + h) * HeadDim + d;
                const uint group_base = (row * Groups + g) * StateSize + s0;
                const float x = static_cast<float>(X[x_idx]);
                const float delta = static_cast<float>(dt[row * Heads + h]);
                const float scale = fast::exp(A * delta);
                float acc = 0.0f;
                for (uint i = 0; i < PerLane; ++i) {
                    const float increment = x * delta * static_cast<float>(B[group_base + i]);
                    const float updated = scale * state[i] + increment;
                    state[i] = updated;
                    acc += updated * C[group_base + i];
                    if (row == 0) {
                        state_confirmed[base + i] = updated;
                    }
                }
                acc = simd_sum(acc);
                if (lane == 0) {
                    out[x_idx] = static_cast<T>(acc + x * D[h]);
                }
            }
            for (uint i = 0; i < PerLane; ++i) {
                state_final[base + i] = state[i];
            }
        """,
    )


def ssm_verify(x, a_log, b, c, d, dt, dt_bias, state, time_step_limit):
    """Return outputs and recurrent state after the confirmed row and final row."""
    from mlx_lm.models.ssm import compute_dt

    _, _, heads, head_dim = x.shape
    groups, state_size = b.shape[-2:]
    return _recurrent_kernel()(
        inputs=[x, a_log, b, c, d, compute_dt(dt, dt_bias, time_step_limit), state],
        template=[
            ("T", x.dtype),
            ("Heads", heads), ("HeadDim", head_dim), ("Groups", groups),
            ("HeadsPerGroup", heads // groups), ("StateSize", state_size),
        ],
        grid=(32, head_dim, heads), threadgroup=(32, 8, 1),
        output_shapes=[x.shape, state.shape, state.shape],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


@cache
def _draft_conv_kernel():
    return mx.fast.metal_kernel(
        name="bolt_lilicorr_grouped_conv_2",
        input_names=["X", "Delta", "Base"],
        output_names=["Y"],
        ensure_row_contiguous=False,
        source="""
            const uint i = thread_position_in_grid.x;
            const uint width = X_shape[1];
            const uint row = i / width;
            if (row >= X_shape[0]) {
                return;
            }
            const uint col = i - row * width;
            const uint group = col / GroupSize;
            const uint here = row * X_strides[0] + col * X_strides[1];
            const uint base_row = col * Base_strides[1];
            const uint param = row * Delta_strides[0] + group * Delta_strides[2];

            // Each stock expression materializes BF16 before the next operation.
            const bfloat c0 = bfloat(float(Base[base_row]) + float(Delta[param]));
            const bfloat c1 = bfloat(float(Base[base_row + Base_strides[0]])
                                 + float(Delta[param + Delta_strides[1]]));
            const bfloat previous = row == 0 ? bfloat(0.0f) : X[here - X_strides[0]];
            const bfloat y0 = bfloat(float(c0) * float(X[here]));
            const bfloat y1 = bfloat(float(c1) * float(previous));
            const bfloat masked = bfloat(float(y1) * float((row % Block) != 0));
            Y[i] = bfloat(float(y0) + float(masked));
        """,
    )


def draft_grouped_conv(x, delta, base, block, group_size):
    """Apply the two-tap LiLiCorr convolution with its original BF16 rounding."""
    return _draft_conv_kernel()(
        inputs=[x, delta, base],
        template=[("Block", block), ("GroupSize", group_size)],
        grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]
