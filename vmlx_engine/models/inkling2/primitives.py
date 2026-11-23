"""Attention, convolution and weight helpers from upstream Inkling."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_MASK_SRC = r"""
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if (i >= LQ || j >= S || bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = (int(i) + int(Q_OFF)) - int(j);   // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        uint rbase = ((b * LQ + i) * H + h) * D_REL;
        uint pcol = (uint)dist;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[((b * H + h) * LQ + i) * S + j] = val;
"""

_mask_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask",
    input_names=["rel", "proj"],
    output_names=["out"],
    source=_MASK_SRC,
)

_MASK_V2_SRC = r"""
    // Shape-generic variant of the banded mask: every runtime dimension comes
    // from injected shapes (B/LQ/H from rel, S from the unread shape-carrier
    // input), and the query offset is S - LQ. Only the per-layer constants
    // (dtype, band geometry) are template args, so exactly one pipeline is
    // compiled per layer kind instead of one per (LQ, S) pair.
    const int B  = rel_shape[0];
    const int LQ = rel_shape[1];
    const int H  = rel_shape[2];
    const int S  = kshape_shape[2];
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = ((int)i + (S - LQ)) - (int)j;     // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0]
            + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0]
                               + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
"""

_mask_v2_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask_v2",
    input_names=["rel", "proj", "kshape"],
    output_names=["out"],
    source=_MASK_V2_SRC,
    ensure_row_contiguous=False,
)

_SCONV_SRC = r"""
    uint c = thread_position_in_grid.x;   // channel
    uint b = thread_position_in_grid.y;   // batch row
    if (c >= C || b >= B) return;
    float w0 = (float)w[c * K + 0];
    float w1 = (float)w[c * K + 1];
    float w2 = (float)w[c * K + 2];
    float w3 = (float)w[c * K + 3];
    // Virtual padded input xp = [state (K-1 rows, fp32); x (L rows, T)].
    for (uint i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int r = (int)(i + k) - (int)(K - 1);  // row into x; negative -> state
            float v = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
            float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
            acc += wk * v;
        }
        // Match the unfused path's rounding: the conv emits bf16 (rounded)
        // before the fp32 residual add; the layer residual is a second bf16
        // add on top (as the decoder layer's x + sconv(r) was).
        float conv_r = (float)((T)acc);
        T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
        if (HAS_RES) {
            out[(b * L + i) * C + c] =
                (T)((float)inner + (float)res[(b * L + i) * C + c]);
        } else {
            out[(b * L + i) * C + c] = inner;
        }
    }
    for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[(b * (K - 1) + s) * C + c] = (r < 0)
            ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
            : (float)x[(b * L + (uint)r) * C + c];
    }
"""

_sconv_kernel = mx.fast.metal_kernel(
    name="inkling_sconv_decode",
    input_names=["x", "state", "w", "res"],
    output_names=["out", "nstate"],
    source=_SCONV_SRC,
)


def _rup(a, m):
    return ((a + m - 1) // m) * m

def banded_additive_mask(
    rel, proj, q_offset, S, sliding, rel_extent, left_padding=None, shape_ref=None
):
    """rel: [B, LQ, H, d_rel]; proj: [d_rel, rel_extent] -> additive mask [B, H, LQ, S]."""
    B, LQ, H, d_rel = rel.shape
    dtype = rel.dtype
    q_offset = int(q_offset)
    S = int(S)
    sliding = int(sliding)
    rel_extent = int(rel_extent)
    if (
        shape_ref is not None
        and shape_ref.ndim >= 3
        and shape_ref.shape[2] == S
        and q_offset == S - LQ
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        mask = _mask_v2_kernel(
            inputs=[rel, proj, shape_ref],
            template=[
                ("T", dtype),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    elif mx.default_device() == mx.gpu and mx.metal.is_available():
        mask = _mask_kernel(
            inputs=[rel, proj],
            template=[
                ("T", dtype),
                ("B", B),
                ("H", H),
                ("LQ", LQ),
                ("S", S),
                ("Q_OFF", q_offset),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    else:
        rl = (rel @ proj).transpose(0, 2, 1, 3)
        qp = mx.arange(LQ) + q_offset
        kp = mx.arange(S)
        dist = qp[:, None] - kp[None, :]
        gidx = mx.broadcast_to(
            mx.clip(dist, 0, rel_extent - 1)[None, None], (B, H, LQ, S)
        )
        pb = mx.take_along_axis(rl, gidx, axis=-1)
        pb = mx.where((dist >= rel_extent)[None, None], mx.array(0.0, dtype), pb)
        masked = dist < 0
        if sliding > 0:
            masked = masked | (dist >= sliding)
        mask = mx.where(masked[None, None], mx.array(-1e30, dtype), pb)

    if left_padding is not None:
        valid = mx.arange(S)[None, :] >= left_padding[:, None]
        mask = mx.where(valid[:, None, None, :], mask, mx.array(-1e30, dtype))
    return mask.astype(dtype)

def _cache_padding_mask(cache, length):
    if cache is None:
        return None
    positions = mx.arange(length)[None, :]
    mask = None
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is not None:
        mask = positions >= left_padding[:, None]
    lengths = getattr(cache, "lengths", None)
    if lengths is not None:
        length_mask = positions < lengths[:, None]
        mask = length_mask if mask is None else mask & length_mask
    return mask

def _next_conv_state(state, inputs, mask):
    state_size = state.shape[1]
    if state_size == 0:
        return state
    if mask is None:
        return mx.concatenate([state, inputs], axis=1)[:, -state_size:]

    valid_count = mx.sum(mask, axis=1).astype(mx.int32)
    valid_start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(state_size)[None, :] + valid_count[:, None] - state_size
    state_indices = state_size + logical
    input_indices = state_size + valid_start[:, None] + mx.maximum(logical, 0)
    indices = mx.where(logical >= 0, input_indices, state_indices)
    combined = mx.concatenate([state, inputs], axis=1)
    indices = mx.broadcast_to(indices[..., None], (*indices.shape, inputs.shape[-1]))
    return mx.take_along_axis(combined, indices, axis=1)

class InklingShortConvolution(nn.Module):
    """Depthwise causal 1-D conv over the previous ``kernel_size - 1`` states, plus a
    residual add. Kept in fp32 for stability (matches the reference). ``conv_idx`` selects
    this conv's slot in the layer's shared conv cache."""

    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        residual: Optional[mx.array] = None,
    ):
        dt = x.dtype
        K = self.kernel_size
        if (
            cache is not None
            and mask is None
            and K == 4
            and x.shape[1] <= 8
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            B, L, C = x.shape
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((B, K - 1, C), dtype=mx.float32)
            out, nstate = _sconv_kernel(
                inputs=[
                    x,
                    state,
                    self.conv.weight.reshape(-1),
                    residual if residual is not None else x,
                ],
                template=[
                    ("T", dt),
                    ("B", B),
                    ("L", L),
                    ("C", C),
                    ("K", K),
                    ("HAS_RES", residual is not None),
                ],
                grid=(_rup(C, 32), B, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, L, C), (B, K - 1, C)],
                output_dtypes=[dt, mx.float32],
            )
            cache[self.conv_idx] = nstate
            return out
        xf = x.astype(mx.float32)
        res = xf
        if mask is not None:
            xf = mx.where(mask[..., None], xf, 0)
        if cache is not None:
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((xf.shape[0], K - 1, xf.shape[-1]), dtype=xf.dtype)
            xp = mx.concatenate([state, xf], axis=1)
            cache[self.conv_idx] = _next_conv_state(state, xf, mask)
        else:
            xp = mx.pad(xf, [(0, 0), (K - 1, 0), (0, 0)])
        out = self.conv(xp.astype(self.conv.weight.dtype)).astype(mx.float32)
        out = (out + res).astype(dt)
        return out if residual is None else residual + out


def masked_scatter(input_tensor, mask, source):
    """MLX port of torch.Tensor.masked_scatter: fill mask==True slots with source in order."""
    mask = mask.astype(mx.bool_)
    if not mask.any():
        return input_tensor
    shape = input_tensor.shape
    flat = input_tensor.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()
    positions = mx.cumsum(mask_flat.astype(mx.int32)) - 1
    positions = mx.clip(positions, 0, source_flat.size - 1)
    selected = source_flat[positions]
    return mx.where(mask_flat, selected, flat).reshape(shape)

def _split_gate_up(v):
    """De-interleave a fused [..., 2*inter, hidden] gate/up weight into (gate, up).

    Return contiguous halves for performance; `gather_mm / `gather_qmm` are much slower when fed strided views
    """
    *lead, two_i, hidden = v.shape
    w = v.reshape(*lead, two_i // 2, 2, hidden)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])

def _to_mlx_conv(v):
    if v.ndim == 3 and v.shape[-1] == 1:
        return v
    return v.transpose(0, 2, 1)
