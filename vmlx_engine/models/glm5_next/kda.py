# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) primitives for glm5_next — vendored.

Port of the parity-proven `jang_tools.ling3.kda` (pinned against the torch
`fla.ops.kda` reference; that package is not shipped inside the product, so
the primitives are vendored here verbatim). The gated delta rule, per head,
with state ``S`` of shape ``[K, V]``:

    D_t = diag(exp(g_t))                     # per-KEY-channel decay
    S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T (q_t * scale)

``q``/``k`` are L2-normalized along the head dim before entering the
recurrence, ``beta = sigmoid(b_proj(x))``, ``scale = K ** -0.5``.

NOTE: glm5_next's decay gate is the smooth sigmoid-bounded form computed in
`glm5_next.py` (`lower_bound * sigmoid(exp(A_log) * (f + dt_bias))`), NOT the
Ling clamped-softplus form — the gate is an input here, never derived here.

⚠️ Do not fuse/refactor the recurrence without re-pinning against the
reference, and test with l2-NORMALIZED q/k: the un-normalized regime is
ill-conditioned and makes a correct kernel look broken (`kda_chunked` shows
deltas up to 70 on unnormalized fixtures while matching to 7e-4 in-regime).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlx.core as mx


_LOGGER = logging.getLogger(__name__)
_EXACT_PAIRWISE_OBSERVED = False
_EXACT_PAIRWISE_REQUESTED = os.environ.get(
    "VMLX_GLM5_EXACT_PAIRWISE_BUFFER", "0"
).strip().lower() not in {"", "0", "false", "off", "no"}

_EXACT_PAIRWISE_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_EXACT_PAIRWISE_SOURCE = """
    size_t output_index = (size_t)thread_position_in_grid.x;
    if (output_index >= (size_t)ELEMENTS) return;

    uint dimension = (uint)(output_index % (size_t)WIDTH);
    size_t pair_index = output_index / (size_t)WIDTH;
    uint column = (uint)(pair_index % (size_t)BLOCK);
    uint row = (uint)((pair_index / (size_t)BLOCK) % (size_t)BLOCK);
    size_t outer = pair_index / ((size_t)BLOCK * (size_t)BLOCK);
    size_t left_index = (outer * (size_t)BLOCK + (size_t)row) *
        (size_t)WIDTH + (size_t)dimension;
    size_t right_index = (outer * (size_t)BLOCK + (size_t)column) *
        (size_t)WIDTH + (size_t)dimension;

    // Match MLX 0.32.2's separate safe-math primitives exactly: Subtract,
    // Minimum, precise::Exp, Multiply, Multiply. The stock mx.sum remains a
    // separate primitive over this row-contiguous product buffer, preserving
    // its reduction tree and accumulation order.
    float gate_delta = float(gates[left_index]) - float(gates[right_index]);
    float clamped_delta = metal::min(gate_delta, 0.0f);
    float decay = metal::precise::exp(clamped_delta);
    float first_product = float(left[left_index]) * decay;
    products[output_index] = first_product * float(right[right_index]);
"""


@lru_cache(maxsize=1)
def _exact_pairwise_product_kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_exact_pairwise_product_v1",
        input_names=["left", "right", "gates"],
        output_names=["products"],
        header=_EXACT_PAIRWISE_HEADER,
        source=_EXACT_PAIRWISE_SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def _exact_pairwise_product(
    left: mx.array,
    right: mx.array,
    gates: mx.array,
) -> mx.array | None:
    """Materialize exact gated products; leave reduction to stock ``mx.sum``."""
    if not _EXACT_PAIRWISE_REQUESTED:
        return None
    if left.ndim != 4 or left.shape != right.shape or left.shape != gates.shape:
        return None
    if any(value.dtype != mx.float32 for value in (left, right, gates)):
        return None
    block = int(left.shape[-2])
    width = int(left.shape[-1])
    if block != 64 or width != 128:
        return None
    outer = 1
    for extent in left.shape[:-2]:
        outer *= int(extent)
    elements = outer * block * block * width
    result = _exact_pairwise_product_kernel()(
        inputs=[left, right, gates],
        template=[
            ("BLOCK", block),
            ("WIDTH", width),
            ("ELEMENTS", elements),
        ],
        grid=(elements, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*left.shape[:-2], block, block, width)],
        output_dtypes=[mx.float32],
    )[0]
    global _EXACT_PAIRWISE_OBSERVED
    if not _EXACT_PAIRWISE_OBSERVED:
        _EXACT_PAIRWISE_OBSERVED = True
        _LOGGER.warning(
            "exact KDA pairwise product buffer engaged: shape=%s dtype=%s",
            tuple(int(value) for value in left.shape),
            left.dtype,
        )
    return result


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """L2-normalize along the last axis, in fp32 (matches the fla kernel)."""
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def kda_recurrent(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Gated delta-rule recurrence.

    q, k: ``[B, T, H, K]`` (already L2-normalized); v: ``[B, T, H, V]``;
    g: ``[B, T, H, K]`` log-space decay (fp32); beta: ``[B, T, H]`` (fp32);
    state: optional ``[B, H, K, V]``. Returns ``(o [B,T,H,V], final_state)``.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q = q.astype(mx.float32) * scale
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)

    S = mx.zeros((B, H, K, V), dtype=mx.float32) if state is None else state.astype(mx.float32)

    decay = mx.exp(g)  # [B, T, H, K]
    out = []
    for t in range(T):
        k_t = k[:, t]
        v_t = v[:, t]
        q_t = q[:, t]
        b_t = beta[:, t]

        S = S * decay[:, t][..., None]     # decay along the key axis
        # delta correction against the *decayed* state (order matters)
        kS = mx.sum(k_t[..., None] * S, axis=-2)          # [B, H, V]
        S = S + (b_t[..., None] * k_t)[..., None] * (v_t - kS)[..., None, :]
        out.append(mx.sum(q_t[..., None] * S, axis=-2))   # [B, H, V]

    return mx.stack(out, axis=1), S


def kda_recurrent_with_states(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    scale: float | None = None,
) -> tuple[mx.array, mx.array, list[mx.array]]:
    """Recurrent KDA plus the exact state after every input position.

    Speculative verification needs an accepted-prefix rollback boundary for
    every draft token.  Returning those states lets the caller keep the QMM,
    gate, convolution, norm, and output projections batched over the verify
    slab instead of rerunning the entire attention block one token at a time.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q = q.astype(mx.float32) * scale
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    S = (
        mx.zeros((B, H, K, V), dtype=mx.float32)
        if state is None
        else state.astype(mx.float32)
    )

    decay = mx.exp(g)
    out = []
    states = []
    for t in range(T):
        k_t = k[:, t]
        v_t = v[:, t]
        q_t = q[:, t]
        b_t = beta[:, t]
        S = S * decay[:, t][..., None]
        kS = mx.sum(k_t[..., None] * S, axis=-2)
        S = S + (b_t[..., None] * k_t)[..., None] * (
            v_t - kS
        )[..., None, :]
        out.append(mx.sum(q_t[..., None] * S, axis=-2))
        states.append(S)

    return mx.stack(out, axis=1), S, states


def kda_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    S: mx.array,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Single-token KDA update — the decode path.

    ``q,k,g: [B,H,K]``, ``v: [B,H,V]``, ``beta: [B,H]``, ``S: [B,H,K,V]``.
    """
    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q.astype(mx.float32) * scale
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    S = S * mx.exp(g.astype(mx.float32))[..., None]
    kS = mx.sum(k[..., None] * S, axis=-2)
    S = S + (beta.astype(mx.float32)[..., None] * k)[..., None] * (v - kS)[..., None, :]
    return mx.sum(q[..., None] * S, axis=-2), S


def short_conv(
    x: mx.array,
    weight: mx.array,
    state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Causal depthwise short convolution with silu (``ShortConvolution``).

    x: ``[B, T, C]``; weight: ``[C, W]`` (bundle stores it squeezed); state:
    optional left context ``[B, W-1, C]``. Returns ``(y [B,T,C], new_state)``.
    """
    if weight.ndim == 3:
        weight = weight.reshape(weight.shape[0], -1)    # [C, 1, W] -> [C, W]
    C, W = weight.shape
    B, T, _ = x.shape

    if state is None:
        state = mx.zeros((B, W - 1, C), dtype=x.dtype)
    padded = mx.concatenate([state.astype(x.dtype), x], axis=1)   # [B, W-1+T, C]

    # depthwise causal conv: y[:, t] = sum_w padded[:, t + w] * weight[:, w]
    y = mx.zeros((B, T, C), dtype=mx.float32)
    for w in range(W):
        y = y + padded[:, w : w + T].astype(mx.float32) * weight[:, w].astype(mx.float32)

    new_state = padded[:, padded.shape[1] - (W - 1):]
    y = y * mx.sigmoid(y)                    # silu
    return y.astype(x.dtype), new_state


def short_conv_with_states(
    x: mx.array,
    weight: mx.array,
    state: mx.array | None = None,
) -> tuple[mx.array, mx.array, list[mx.array]]:
    """Causal short convolution plus its tail after every input position."""
    if weight.ndim == 3:
        weight = weight.reshape(weight.shape[0], -1)
    C, W = weight.shape
    B, T, _ = x.shape
    if state is None:
        state = mx.zeros((B, W - 1, C), dtype=x.dtype)
    padded = mx.concatenate([state.astype(x.dtype), x], axis=1)

    y = mx.zeros((B, T, C), dtype=mx.float32)
    for w in range(W):
        y = y + padded[:, w : w + T].astype(mx.float32) * weight[
            :, w
        ].astype(mx.float32)

    states = [padded[:, t + 1 : t + W] for t in range(T)]
    y = y * mx.sigmoid(y)
    return y.astype(x.dtype), states[-1], states


def kda_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """Chunked (WY-form) gated delta rule — the T>64 prefill path.

    Port of `fla.ops.kda.naive.naive_chunk_kda` (H == HV) with the two
    per-token inner loops vectorized via broadcasting. Pinned against
    :func:`kda_recurrent` at non-aligned lengths. `T` is padded internally to
    a multiple of `chunk_size` with beta=0 / k=0 rows (state-neutral by
    construction); output is sliced back.

    Memory note: Akk is built PER CHUNK — the all-chunks broadcast is
    ~19 GB fp32/layer at a 4.5k-token prefill (measured swapping a 128 GB
    box); per-chunk slices peak at ~270 MB.
    """
    B, T0, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    BT = chunk_size
    pad = (-T0) % BT
    if pad:
        q = mx.concatenate([q, mx.zeros((B, pad, H, K), q.dtype)], axis=1)
        k = mx.concatenate([k, mx.zeros((B, pad, H, K), k.dtype)], axis=1)
        v = mx.concatenate([v, mx.zeros((B, pad, H, V), v.dtype)], axis=1)
        g = mx.concatenate([g, mx.zeros((B, pad, H, K), g.dtype)], axis=1)
        beta = mx.concatenate([beta, mx.zeros((B, pad, H), beta.dtype)], axis=1)
    T = T0 + pad
    NT = T // BT

    def chunked(x, d):
        # [B, T, H, d] -> [B, H, NT, BT, d]
        return x.reshape(B, NT, BT, H, d).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    qc = chunked(q, K) * scale
    kc = chunked(k, K)
    vc = chunked(v, V)
    gc = mx.cumsum(chunked(g, K), axis=-2)                 # within-chunk cumsum
    bc = beta.reshape(B, NT, BT, H).transpose(0, 3, 1, 2).astype(mx.float32)

    # Akk[..., c, j] = sum_d k[c,d] * exp(g[c,d]-g[j,d]) * k[j,d] for c > j.
    # Clamp BEFORE exp: lossless on the used lower-triangular region (g is a
    # cumsum of non-positives); the upper region would produce inf otherwise.
    lower = mx.tril(mx.ones((BT, BT), dtype=mx.bool_), k=-1)
    akk_chunks = []
    for ci in range(NT):
        g_c = gc[:, :, ci]                                  # [B,H,BT,K]
        k_c = kc[:, :, ci]
        products = _exact_pairwise_product(k_c, k_c, g_c)
        if products is None:
            gd_c = mx.minimum(
                g_c[..., :, None, :] - g_c[..., None, :, :], 0.0
            )
            products = (
                k_c[..., :, None, :] * mx.exp(gd_c)
                * k_c[..., None, :, :]
            )
        akk_chunks.append(mx.sum(products, axis=-1))
    Akk = mx.stack(akk_chunks, axis=2)                      # [B,H,NT,BT,BT]
    A = mx.where(lower, -(Akk * bc[..., None]), mx.zeros_like(Akk))

    # forward substitution: (I - lower(A))^{-1}-style accumulation
    for i in range(1, BT):
        upd = mx.sum(A[..., i, :, None] * A[..., :, :i], axis=-2)
        A[..., i, :i] = A[..., i, :i] + upd
    A = (A + mx.eye(BT, dtype=mx.float32)) * bc[..., None, :]

    w = A @ (mx.exp(gc) * kc)                               # [B,H,NT,BT,K]
    u = A @ vc                                              # [B,H,NT,BT,V]

    S = mx.zeros((B, H, K, V), dtype=mx.float32) if state is None else state.astype(mx.float32)
    strict_upper = mx.triu(mx.ones((BT, BT), dtype=mx.bool_), k=1)
    outs = []
    for i in range(NT):
        q_i, k_i = qc[:, :, i], kc[:, :, i]
        u_i, g_i, w_i = u[:, :, i], gc[:, :, i], w[:, :, i]
        products = _exact_pairwise_product(q_i, k_i, g_i)
        if products is None:
            gd = mx.minimum(
                g_i[..., :, None, :] - g_i[..., None, :, :], 0.0
            )
            products = (
                q_i[..., :, None, :] * mx.exp(gd)
                * k_i[..., None, :, :]
            )
        Aqk = mx.sum(products, axis=-1)
        Aqk = mx.where(strict_upper, mx.zeros_like(Aqk), Aqk)
        v_i = u_i - w_i @ S
        outs.append((q_i * mx.exp(g_i)) @ S + Aqk @ v_i)
        g_last = g_i[:, :, -1]                              # [B,H,K]
        S = S * mx.exp(g_last)[..., None]
        S = S + ((mx.exp(g_last[:, :, None, :] - g_i) * k_i).transpose(0, 1, 3, 2) @ v_i)

    o = mx.stack(outs, axis=2)                              # [B,H,NT,BT,V]
    o = o.transpose(0, 2, 3, 1, 4).reshape(B, T, H, V)
    return o[:, :T0], S
