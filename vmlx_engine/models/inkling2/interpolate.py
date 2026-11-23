"""Inkling vision positional resize from the pinned MLX-VLM source."""

import math
from functools import lru_cache

import mlx.core as mx

_HAS_METAL = mx.metal.is_available()


def _separable_interpolate_axis(x, idx, w):
    N, _, M = x.shape
    out, taps = idx.shape
    g = mx.take(x, idx.reshape(-1), axis=1).reshape(N, out, taps, M)
    return mx.sum(g * w[None, :, :, None], axis=2)

def _separable_interpolate_mlx(x, iy, wy, ix, wx):
    """Pure-MLX separable interpolation (no Metal kernel)."""
    N, H, W, C = x.shape
    out_h, taps_y = iy.shape
    out_w, taps_x = ix.shape
    # Both axes are gathered along axis 1 so the tap reduction keeps a long inner axis
    h_first_cost = out_h * W * (taps_y + 1) + out_h * out_w * (taps_x + 1)
    w_first_cost = H * W + H * out_w * (taps_x + 1) + out_h * out_w * taps_y
    if h_first_cost <= w_first_cost:
        y = _separable_interpolate_axis(x.reshape(N, H, W * C), iy, wy)
        y = y.reshape(N, out_h, W, C).transpose(0, 2, 1, 3).reshape(N, W, out_h * C)
        out = _separable_interpolate_axis(y, ix, wx).reshape(N, out_w, out_h, C)
        return out.transpose(0, 2, 1, 3)
    y = x.transpose(0, 2, 1, 3).reshape(N, W, H * C)
    y = _separable_interpolate_axis(y, ix, wx)
    y = y.reshape(N, out_w, H, C).transpose(0, 2, 1, 3).reshape(N, H, out_w * C)
    return _separable_interpolate_axis(y, iy, wy).reshape(N, out_h, out_w, C)

def separable_interpolate(x, iy, wy, ix, wx, *, training=False):
    """
    Separable interpolation of a channel-last tensor from per-axis tap tables.

    Args:
        x: MLX tensor of shape [B, H, W, C]
        iy, ix: int32 source indices of shape [out_h, taps_y] and [out_w, taps_x]
        wy, wx: float32 weights with the same shapes as iy and ix
        training: Use differentiable MLX operations instead of the Metal kernel.

    Returns:
        float32 tensor of shape [B, out_h, out_w, C]
    """
    if training or not _HAS_METAL:
        return _separable_interpolate_mlx(x, iy, wy, ix, wx)

    batch_size, _, _, channels = x.shape
    out_h = iy.shape[0]
    out_w = ix.shape[0]

    source = """
        uint gx = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint b = thread_position_in_grid.z;

        int in_h = x_shape[1];
        int in_w = x_shape[2];
        int channels = x_shape[3];
        int out_h = iy_shape[0];
        int taps_y = iy_shape[1];
        int out_w = ix_shape[0];
        int taps_x = ix_shape[1];

        if (gx >= (uint)(out_w * channels) || y_out >= (uint)out_h)
            return;

        int x_out = gx / channels;
        size_t y_tap = (size_t)y_out * taps_y;
        size_t x_tap = (size_t)x_out * taps_x;
        size_t input_base = (size_t)b * in_h * in_w * channels + (gx % channels);

        // W taps are contracted inside the H loop to match ATen's summation order
        float result = 0.0f;
        for (int a = 0; a < taps_y; a++) {
            size_t row = input_base + (size_t)iy[y_tap + a] * in_w * channels;
            float row_result = 0.0f;
            for (int c = 0; c < taps_x; c++) {
                row_result += x[row + (size_t)ix[x_tap + c] * channels] * wx[x_tap + c];
            }
            result += row_result * wy[y_tap + a];
        }

        out[((size_t)b * out_h + y_out) * out_w * channels + gx] = result;
    """

    kernel = mx.fast.metal_kernel(
        name="separable_interpolate",
        input_names=["x", "iy", "wy", "ix", "wx"],
        output_names=["out"],
        source=source,
    )

    outputs = kernel(
        inputs=[x.astype(mx.float32), iy, wy, ix, wx],
        grid=(out_w * channels, out_h, batch_size),
        threadgroup=(min(256, out_w * channels), 1, 1),
        output_shapes=[(batch_size, out_h, out_w, channels)],
        output_dtypes=[mx.float32],
    )

    return outputs[0]


@lru_cache(maxsize=64)
def _bilinear_weights_1d(in_size, out_size, align_corners=False, antialias=False):
    """(out, taps) indices and weights matching ATen ``upsample_bilinear2d``
    and, for antialiased downscales, ``_upsample_bilinear2d_aa``."""
    if align_corners:
        scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
        src = mx.arange(out_size, dtype=mx.float32) * scale
    else:
        scale = in_size / out_size
        src = (mx.arange(out_size, dtype=mx.float32) + 0.5) * scale - 0.5
    if antialias and scale > 1.0:
        lo = mx.maximum(mx.floor(src - scale + 1), 0)
        hi = mx.minimum(mx.floor(src + scale), in_size - 1)
        # hi - lo + 1 <= ceil(2 * scale) for every pixel, so the tap count is static
        n_taps = math.ceil(2 * scale)
        pos = lo[:, None] + mx.arange(n_taps, dtype=mx.float32)[None, :]
        w = mx.maximum(0.0, 1.0 - mx.abs(pos - src[:, None]) / scale)
        w = mx.where(pos <= hi[:, None], w, 0.0)
        w = w / w.sum(axis=1, keepdims=True)
        # Zero-weight taps past hi are clamped so the gather stays in bounds
        idx = mx.minimum(pos, in_size - 1).astype(mx.int32)
    else:
        src = mx.maximum(src, 0.0)
        base = mx.floor(src)
        frac = src - base
        pos = base[:, None] + mx.arange(2, dtype=mx.float32)[None, :]
        idx = mx.clip(pos.astype(mx.int32), 0, in_size - 1)
        w = mx.stack([1.0 - frac, frac], axis=1)
    return idx, w

def resize_bilinear_nhwc(
    x, new_size, align_corners=False, antialias=False, *, training=False
):
    """Channel-last bilinear resize matching ``F.interpolate``; ``antialias``
    is PyTorch's triangle filter rather than ``resize_bilinear``'s Gaussian blur.
    Set ``training=True`` to use differentiable MLX operations."""
    _, H, W, _ = x.shape
    new_height, new_width = new_size
    if (H, W) == (new_height, new_width):
        return x
    iy, wy = _bilinear_weights_1d(H, new_height, align_corners, antialias)
    ix, wx = _bilinear_weights_1d(W, new_width, align_corners, antialias)
    return separable_interpolate(x, iy, wy, ix, wx, training=training)
