"""Blocked sliding-window prefill attention for Naive N0.5 SWA layers (vMLX Python, INTERNAL, 2026-10-09).

Prefill with window W ran ONE SDPA of L queries over all L + W - 1 cached keys under a window mask (12.6 ms per
2048-token chunk-layer on M5 Max, ~94% of the score tiles masked out). Exact rewrite: split queries into blocks of B = W;
block b only needs keys [bW - W, bW + W) -> a 2W-key slab = [previous block | own block] of the front-padded key array.
All blocks run as ONE batched SDPA (batch = block) with the same (B, 2B) window mask, the attention sink bias and GQA.
Score tensor = nb x H x W x 2W (small: far below MLX's >8 GiB sinks bug).
"""
from __future__ import annotations

import mlx.core as mx


def swa_blocked_attention(q, k, v, scale: float, sinks, window: int):
    """q (1, H, L, D) at the last L positions of the keys; k/v (1, KV, Lk, .) temporal order, Lk >= L.
    Returns (1, H, L, Dv). Exact (same masked softmax incl. sink column) as the dense window-masked SDPA."""
    _, H, L, D = q.shape
    W = window
    hist = min(k.shape[2] - L, W - 1)                     # history keys a query can still see
    k, v = k[:, :, k.shape[2] - (L + hist):], v[:, :, v.shape[2] - (L + hist):]
    nb = -(-L // W); Lp = nb * W
    # front pad to W history slots, back pad queries/keys to whole blocks
    kp = mx.pad(k, [(0, 0), (0, 0), (W - hist, Lp - L), (0, 0)])
    vp = mx.pad(v, [(0, 0), (0, 0), (W - hist, Lp - L), (0, 0)])
    qp = mx.pad(q, [(0, 0), (0, 0), (0, Lp - L), (0, 0)]) if Lp != L else q
    KV = k.shape[1]
    def slab(t):                                           # (1, KV, W + Lp, d) -> (nb, KV, 2W, d)
        a = t[:, :, :Lp].reshape(KV, nb, W, -1); b = t[:, :, W:W + Lp].reshape(KV, nb, W, -1)
        return mx.concatenate([a, b], axis=2).transpose(1, 0, 2, 3)
    ks, vs = slab(kp), slab(vp)
    qs = qp.reshape(H, nb, W, D).transpose(1, 0, 2, 3)     # (nb, H, W, D)
    i = mx.arange(W)[:, None]; t = mx.arange(2 * W)[None, :]
    d = (i + W) - t                                        # query padded index - key padded index (within the slab)
    win = (d >= 0) & (d < W)                               # (W, 2W)
    valid = (mx.arange(nb)[:, None] * W + mx.arange(2 * W)[None, :]) >= (W - hist)   # (nb, 2W): front padding invalid
    mask = (win[None] & valid[:, None, :])[:, None]        # (nb, 1, W, 2W)
    o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=scale, mask=mask, sinks=sinks)   # (nb, H, W, Dv)
    return o.transpose(1, 0, 2, 3).reshape(1, H, Lp, -1)[:, :, :L]
